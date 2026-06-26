// panels.js: one introspection readout per side. Buttons fired are shown first, then:
//   - a code policy: its real Python with the line(s) executing this frame lit;
//   - the RNN baseline: the observation in, the recurrent state in, and the logits out (the latent state =
//     pre-threshold tanh outputs; the first 3 cross the threshold to become the buttons);
//   - a human-driven side: the buttons being pressed.
// createPanel(el) returns { update(view) }.
//
// view = { mode:'code'|'rnn'|'human', label, source, loading,
//          firedLines:Set, acts:{obs,state,logits}, buttons:[f,b,j] }

const KEYWORDS = new Set('def return if elif else for while in and or not import from as None True False break continue lambda class'.split(' '));
const BUILTINS = new Set(['np', 'Math']);
const EMPTY = new Set();
const EMPTY_ARR = [];

function escHtml(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

// line-based Python highlighter (our sources have no multi-line tokens)
function highlight(line) {
  let code = line, comment = '', inStr = false, q = '';
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) { if (ch === q) inStr = false; continue; }
    if (ch === "'" || ch === '"' || ch === '`') { inStr = true; q = ch; continue; }
    if (ch === '#') { code = line.slice(0, i); comment = line.slice(i); break; }
  }
  const re = /('(?:\\.|[^'])*'|"(?:\\.|[^"])*")|(\b\d+\.?\d*\b)|([A-Za-z_]\w*)|(\s+)|([^\sA-Za-z0-9_])/g;
  let html = '', m;
  while ((m = re.exec(code))) {
    if (m[1] !== undefined) html += `<span class="tk-s">${escHtml(m[1])}</span>`;
    else if (m[2] !== undefined) html += `<span class="tk-n">${m[2]}</span>`;
    else if (m[3] !== undefined) html += KEYWORDS.has(m[3]) ? `<span class="tk-k">${m[3]}</span>` : (BUILTINS.has(m[3]) ? `<span class="tk-b">${m[3]}</span>` : escHtml(m[3]));
    else html += escHtml(m[4] !== undefined ? m[4] : m[5]);
  }
  if (comment) html += `<span class="tk-c">${escHtml(comment)}</span>`;
  return html;
}

export function createPanel(el) {
  let mode = null, label = null, source = null, loading = null;
  let lineEls = null, obsCells = null, stateCells = null, logitCells = null, buttonCells = null;

  const ttl = text => { const d = document.createElement('div'); d.className = 'ttl'; d.textContent = text; el.appendChild(d); };
  const note = text => { const d = document.createElement('div'); d.className = 'pnote'; d.textContent = text; el.appendChild(d); };
  function cells(n) {
    const row = document.createElement('div'); row.className = 'units'; const out = [];
    for (let i = 0; i < n; i++) { const u = document.createElement('span'); u.className = 'u'; row.appendChild(u); out.push(u); }
    el.appendChild(row); return out;
  }
  function buttons(labelText) {
    ttl(labelText);
    const row = document.createElement('div'); row.className = 'units';
    const out = ['fwd', 'back', 'jump'].map(t => { const b = document.createElement('span'); b.className = 'u btn3'; b.textContent = t; row.appendChild(b); return b; });
    el.appendChild(row); return out;
  }
  function legend() {
    const d = document.createElement('div'); d.className = 'legend';
    d.innerHTML = '<span class="lk obs"></span>observation<span class="lk state"></span>recurrent state<span class="lk logit"></span>logits';
    el.appendChild(d);
  }

  function rebuild(view) {
    el.innerHTML = ''; lineEls = obsCells = stateCells = logitCells = buttonCells = null;
    if (view.mode === 'human') {
      buttonCells = buttons('buttons you press');
      ttl('▸ you’re driving the magenta slime :-D');
      note('Reselect the magenta model above (or press Q) to hand this slime back to a policy.');
    } else if (view.mode === 'code') {
      buttonCells = buttons('buttons fired');
      ttl(view.label + ' · its real Python, running in your browser' + (view.loading ? ' (loading…)' : (view.loadError ? ' (can’t run in browser)' : '')));
      const code = document.createElement('div'); code.className = 'code';
      const inner = document.createElement('div'); inner.className = 'codeinner';
      lineEls = (view.source || '').split('\n').map((raw, i) => { const d = document.createElement('div'); d.className = 'cline'; d.innerHTML = highlight(raw) || '&nbsp;'; inner.appendChild(d); return { el: d, n: i + 1 }; });
      code.appendChild(inner); el.appendChild(code);
      note('Runs unchanged in-browser via Pyodide (CPython→WASM). The highlighted lines are the ones executing each frame.');
    } else if (view.mode === 'rnn') {
      buttonCells = buttons('buttons fired');
      ttl(view.label + ' · recurrent net (120 params)');
      legend();
      ttl('observation in (8)'); obsCells = cells(8);
      ttl('recurrent state in (7)'); stateCells = cells(7);
      ttl('logits out (7) · the latent state; first 3 → buttons'); logitCells = cells(7);
    }
    mode = view.mode; label = view.label; source = view.source; loading = view.loading;
  }

  // colour a cell row by value: opacity ~ |value|; `signed` shows sign (mint = +, slate = −).
  function paint(arr, vals, rgb, signed) {
    for (let i = 0; i < arr.length; i++) {
      const v = vals[i] || 0, a = (0.12 + 0.85 * Math.min(1, Math.abs(v))).toFixed(3);
      arr[i].style.background = `rgba(${signed && v < 0 ? '120,135,150' : rgb},${a})`;
    }
  }

  function update(view) {
    if (view.mode !== mode || view.label !== label || view.source !== source || view.loading !== loading) rebuild(view);
    const b = view.buttons || EMPTY_ARR;
    if (buttonCells) for (let i = 0; i < 3; i++) buttonCells[i].classList.toggle('on', !!b[i]);
    if (mode === 'code' && lineEls) {
      const lit = view.firedLines || EMPTY;
      for (const ln of lineEls) ln.el.classList.toggle('on', lit.has(ln.n));
    } else if (mode === 'rnn' && logitCells) {
      const a = view.acts || { obs: EMPTY_ARR, state: EMPTY_ARR, logits: EMPTY_ARR };
      paint(obsCells, a.obs, '64,108,150', false);     // observation (blue)
      paint(stateCells, a.state, '192,125,30', false);  // recurrent state (amber)
      paint(logitCells, a.logits, '15,158,133', true);  // logits: mint (+) / slate (−)
    }
  }

  return { update };
}
