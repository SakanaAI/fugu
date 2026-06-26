// pyrunner.js: runs each policy's real Python in-browser via Pyodide.
//
// Boots Pyodide (from its CDN) once, runs a policy's .py as the actor, and captures the per-frame firing-line
// (sys.settrace) that drives source-highlighting. A small obs marshaller turns the 12-d /10 side-relative
// observation into the numpy vector the policy expects; act() returns [forward, backward, jump].

const CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';

// installed once in the interpreter: the policy-entry resolver + the settrace line-tracer.
export const RUNTIME_SRC = `
import sys as _ap_sys, types as _ap_t
import numpy as _np

def _autoport_make(mod, seat):
    pol=None
    f=getattr(mod,"make_policy",None)
    if callable(f):
        try: pol=f(seat=seat)
        except TypeError: pol=f()
    if pol is None:
        if callable(getattr(mod,"act",None)): pol=mod.act
        elif isinstance(getattr(mod,"Policy",None),type): pol=mod.Policy()
    if pol is None: raise AttributeError("no make_policy/act/Policy")
    if callable(pol) and not hasattr(pol,"act"): return pol
    a=getattr(pol,"act",None)
    if callable(a): return a
    if callable(pol): return pol
    raise AttributeError("policy result not callable")

_AP_TRACE = {}
_AP_FILES = set()
def _ap_register(fn):
    _AP_FILES.add(fn)
def _ap_clear(fn):
    _AP_TRACE[fn] = set()
def _ap_get(fn):
    return sorted(_AP_TRACE.get(fn, ()))
def _ap_tracer(frame, event, arg):
    if event == "line":
        fn = frame.f_code.co_filename
        if fn in _AP_FILES:
            s = _AP_TRACE.get(fn)
            if s is None:
                s = set(); _AP_TRACE[fn] = s
            s.add(frame.f_lineno)
    return _ap_tracer

def _slime_obs(o):
    return _np.asarray(o, dtype=_np.float64)   # a Python list of 12 floats -> the numpy vector the policy expects
`;

let _pyPromise = null, _counter = 0;

// Load Pyodide + numpy + the runtime, once. In the browser, loadPyodide is pulled from the CDN <script>.
export function bootPy() {
  if (_pyPromise) return _pyPromise;
  _pyPromise = (async () => {
    if (typeof loadPyodide !== 'function') {
      await new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = CDN + 'pyodide.js'; s.onload = res;
        s.onerror = () => rej(new Error('failed to load pyodide.js from CDN'));
        document.head.appendChild(s);
      });
    }
    const py = await loadPyodide({ indexURL: CDN });
    await py.loadPackage('numpy');
    py.runPython(RUNTIME_SRC);
    return py;
  })();
  return _pyPromise;
}

// Compile one policy's source; returns { ok, factory? }. factory() -> a fresh actor:
//   { act(obs12) -> [f,b,j], reset(), firedLines() -> Set<lineno> }
export async function makePyFactory(source, name) {
  let py;
  try { py = await bootPy(); } catch (e) { return { ok: false, reason: 'pyodide boot failed: ' + (e.message || e) }; }

  const cnt = _counter++;
  const ns = `__slime_${cnt}`;
  const fname = String(name || 'policy').replace(/[^A-Za-z0-9_]/g, '_') + '__' + cnt + '.py';   // settrace matches this
  const remake = py.runPython('(lambda ns: _autoport_make(globals()[ns], 0))');                 // a fresh seat-0 actor
  try {
    py.runPython(
      `import types as _t\n${ns} = _t.ModuleType(${JSON.stringify(ns)})\n`
      + `exec(compile(${JSON.stringify(source)}, ${JSON.stringify(fname)}, "exec"), ${ns}.__dict__)\n`,
    );
    const probe = remake(ns); probe.destroy && probe.destroy();   // confirm the policy's entry point resolves
  } catch (e) {
    return { ok: false, reason: 'policy load failed: ' + String(e.message || e).split('\n')[0] };
  }

  const apClear = py.globals.get('_ap_clear');
  const apGet = py.globals.get('_ap_get');
  py.globals.get('_ap_register')(fname);
  const mkObs = py.globals.get('_slime_obs');
  const traceOn = py.runPython('lambda: __import__("sys").settrace(_ap_tracer)');
  const traceOff = py.runPython('lambda: __import__("sys").settrace(None)');
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  const factory = () => {
    let act = remake(ns);
    // adaptive firing-line: trace every frame for cheap policies; auto-throttle a heavy one (claude's MPC)
    // so tracing never tanks the framerate; its highlight just refreshes less often.
    let every = 1, since = 99, warm = 0;
    return {
      _lines: new Set(),
      act(obs12) {
        const lst = py.toPy(obs12);          // JS array -> Python list
        const arg = mkObs(lst); lst.destroy();
        const traced = (++since >= every);
        if (traced) { since = 0; apClear(fname); traceOn(); }
        const t0 = now();
        let r;
        try { r = act(arg); } finally { if (traced) traceOff(); arg.destroy && arg.destroy(); }
        const dt = now() - t0;
        if (traced) {
          const lj = apGet(fname);
          this._lines = new Set(Array.from(lj.toJs ? lj.toJs() : lj));
          lj.destroy && lj.destroy();
        }
        if (warm < 3) warm++; else if (dt > 12) every = Math.min(every + 1, 30);   // throttle expensive policies
        const out = (r && r.toJs) ? Array.from(r.toJs()) : Array.from(r);
        r && r.destroy && r.destroy();
        return [out[0] > 0 ? 1 : 0, out[1] > 0 ? 1 : 0, out[2] > 0 ? 1 : 0];
      },
      reset() { act && act.destroy && act.destroy(); act = remake(ns); },
      firedLines() { return this._lines; },
    };
  };
  return { ok: true, factory };
}
