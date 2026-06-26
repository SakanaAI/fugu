"""
Fugu use-case: Rubik's cube solver comparison across LLM models.

Each model gets the SAME prompt and must return a self-contained Python module
defining `solve(facelet) -> str`: a general Rubik's cube solver that returns a
WCA move sequence, written from scratch (no off-the-shelf solver library). The
model spends tokens ONCE (writing the solver); we then run that solver locally
against the frozen eval set (eval_cubes.json, 300 cubes) -- the solving itself
costs no tokens.

Every returned solution is re-verified with our trusted engine (cube.py); a
model cannot win by claiming a cube is solved. We compare solve-rate and turn
count (HTM), with a "hero" cube (#0) for the visual/story and the full set for
a robust average.

NOTE: this script CALLS each model's API (needs the relevant SDKs + keys). To
reproduce the published numbers WITHOUT any API call, use rerun_saved.py on the
frozen results/<key>_solver.py files instead.

Run:  python3 rubik_compare.py            # all models
      python3 rubik_compare.py fugu_ultra  # re-run a subset
"""
import os
import sys
import json
import time
import textwrap
import statistics
import threading
import subprocess
import concurrent.futures
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)
EVAL_CUBES = HERE / "eval_cubes.json"
SOLVER_RUNNER = HERE / "solver_runner.py"

# Per-cube wall cap for a model's solver (a table build / optimal search may be
# slow; generous so we never cut off a legitimate solver). Overridable via env.
PER_CUBE_TIMEOUT = float(os.environ.get("RUBIK_CUBE_TIMEOUT", "180"))
API_TIMEOUT = 8000          # reasoning models are slow; do not cut them off

sys.path.insert(0, str(HERE))
from cube import verify_solution, count_turns   # noqa: E402  (trusted engine)

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


PROMPT = textwrap.dedent("""\
    Write a complete, self-contained Python module that solves a Rubik's cube.
    It must define exactly this function:

        def solve(facelet: str) -> str

    which takes a scrambled cube as a 54-character facelet string and returns a
    sequence of moves that solves it, as a single space-separated string.

    FACELET STRING FORMAT (input)
      54 characters, the six faces concatenated in this order: U R F D L B.
      Each face is 9 characters read row-major (top-left to bottom-right) in the
      standard unfolded-net orientation. Each character is the face letter whose
      colour that facelet shows (one of U R F D L B). Index ranges:
        U = 0..8   R = 9..17   F = 18..26   D = 27..35   L = 36..44   B = 45..53
      A SOLVED cube is therefore:
        UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
      Centres never move, so the centre of each face (indices 4,13,22,31,40,49)
      always equals that face's solved colour.

    MOVE NOTATION (output)
      Use standard WCA notation. A bare letter (U D L R F B) is a 90-degree
      CLOCKWISE turn of that face viewed from OUTSIDE the cube. A prime (U')
      is counter-clockwise. A 2 (U2) is a 180-degree turn. Return the moves as
      one space-separated string, e.g.  "R U R' U2 F D' L2".
      An empty string means "already solved".

    OBJECTIVE
      Return a CORRECT solution using AS FEW TURNS AS POSSIBLE (every token --
      X, X', or X2 -- counts as one turn). Any cube is solvable in at most 20
      turns (God's number), so aim low. Correctness first, then minimise turns.

    CONSTRAINTS
      - Use ONLY the Python standard library. Do NOT import numpy or any
        third-party package, and in particular NO Rubik's cube / solver library
        (no kociemba, pycuber, twophase, RubikTwoPhase, magiccube, etc.).
        Implement the cube model and the solving algorithm yourself.
      - solve() must be a pure function of its argument: do not read stdin,
        argv, files, or the network. We import your module and call solve()
        directly, once per cube, on many different scrambles.
      - It should be reasonably fast (we cap each solve at a wall-clock limit).

    Output ONLY the Python code. No explanation, no markdown code fences.
""")

def call_gpt55():
    import openai
    client = openai.OpenAI(timeout=API_TIMEOUT, max_retries=0)
    resp = client.responses.create(
        model="gpt-5.5",
        input=PROMPT,
        reasoning={"effort": "high"},        
        max_output_tokens=64000,
    )
    return resp.output_text


def call_opus48():
    import anthropic
    client = anthropic.Anthropic(timeout=API_TIMEOUT, max_retries=0)
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},    
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for _ in stream:
            pass
    resp = stream.get_final_message()
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def call_fable5():
    import anthropic
    client = anthropic.Anthropic(timeout=API_TIMEOUT, max_retries=0)
    with client.messages.stream(
        model="claude-fable-5",
        max_tokens=128000,                   # 64k starved at xhigh; Fable 5 supports 128k via stream
        thinking={"type": "adaptive"},
        output_config={"effort": "xhigh"},   # high gave mean 20.15; testing xhigh
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for _ in stream:
            pass
    resp = stream.get_final_message()
    for block in resp.content:
        if block.type == "text":
            return block.text
    return ""


def call_gemini():
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=API_TIMEOUT * 1000),  # ms
    )
    resp = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=PROMPT,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH
            ),
            max_output_tokens=65536,
        ),
    )
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None) and not getattr(part, "thought", False):
            return part.text
    return resp.text or ""


def call_fugu():
    import openai
    client = openai.OpenAI(
        api_key=os.environ["FUGU_API_KEY"],
        base_url="https://api.sakana.ai/v1",
        timeout=API_TIMEOUT, max_retries=0,
    )
    resp = client.chat.completions.create(
        model="fugu",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.7,
        max_tokens=64000,
    )
    return resp.choices[0].message.content


def call_fugu_ultra():
    import openai
    client = openai.OpenAI(
        api_key=os.environ["FUGU_API_KEY"],
        base_url="https://api.sakana.ai/v1",
        timeout=API_TIMEOUT, max_retries=0,
    )
    resp = client.chat.completions.create(
        model="fugu-ultra",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=64000,
        extra_body={"reasoning_effort": "max"},   # fugu-ultra accepts only high|max
        stream=True,
        stream_options={"include_usage": True},
    )
    text = ""
    for chunk in resp:
        if getattr(chunk, "choices", None):
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                text += piece
    return text


# ---------------------------------------------------------------------------
def extract_code(raw: str) -> str:
    import re
    text = (raw or "").strip()
    blocks = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else text


def run_solver(name: str, code: str, cubes):
    """Run a model's solver over all cubes in a subprocess. Returns a per-cube
    list of verified results: {id, solved, turns, solution, error, secs}."""
    solver_file = OUT_DIR / f"{name}_solver.py"
    solver_file.write_text(code, encoding="utf-8")

    overall = PER_CUBE_TIMEOUT * len(cubes) + 120
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(SOLVER_RUNNER),
             str(solver_file), str(EVAL_CUBES), str(PER_CUBE_TIMEOUT)],
            capture_output=True, text=True, timeout=overall,
        )
    except subprocess.TimeoutExpired:
        return None, f"solver subprocess exceeded {overall:g}s overall"

    raw_out = {}
    contract_fail = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("##R## "):
            rec = json.loads(line[len("##R## "):])
            raw_out[rec["id"]] = rec
        elif line.startswith("##CONTRACT_FAIL## "):
            contract_fail = line[len("##CONTRACT_FAIL## "):]
    if contract_fail:
        return None, f"contract fail: {contract_fail}"
    if not raw_out:
        err_tail = " | ".join((proc.stderr or "").strip().splitlines()[-2:])
        return None, f"no results (stderr: {err_tail[:200]})"

    results = []
    for cube in cubes:
        rec = raw_out.get(cube["id"], {"solution": None, "error": "missing", "secs": 0.0})
        solved, turns, err = False, None, rec.get("error")
        if rec.get("solution") is not None and err is None:
            try:
                solved, turns = verify_solution(cube["facelet"], rec["solution"])
                if not solved:
                    err = "wrong (did not reach solved)"
                    turns = None
            except Exception as exc:
                err = f"bad moves: {type(exc).__name__}: {str(exc)[:120]}"
        results.append({
            "id": cube["id"], "hero": cube["hero"], "solved": solved,
            "turns": turns, "solution": rec.get("solution"),
            "error": err, "secs": rec.get("secs", 0.0),
        })
    return results, None


# ---------------------------------------------------------------------------
MODELS = [
    ("gpt55",      "GPT-5.5",                call_gpt55),
    ("opus48",     "Opus 4.8",               call_opus48),
    ("fable5",     "Fable 5",                call_fable5),
    ("gemini",     "Gemini 3.1 Pro Preview", call_gemini),
    ("fugu",  "Fugu",              call_fugu),
    ("fugu_ultra", "Fugu-ultra",             call_fugu_ultra),
]


def process_model(key, label, call_fn, cubes):
    t0 = time.time()
    log(f"[{key}] calling API ({label})...")
    try:
        raw = call_fn()
    except Exception as exc:
        log(f"[{key}] API FAILED ({time.time()-t0:.0f}s): {type(exc).__name__}: {str(exc)[:160]}")
        return {"key": key, "label": label, "ok": False, "stage": "api",
                "detail": f"{type(exc).__name__}: {str(exc)[:160]}"}
    api_sec = time.time() - t0
    code = extract_code(raw)
    (OUT_DIR / f"{key}_raw.txt").write_text(raw or "", encoding="utf-8")
    log(f"[{key}] API done ({api_sec:.0f}s), {len(code)} chars; running solver on {len(cubes)} cubes...")

    t1 = time.time()
    results, err = run_solver(key, code, cubes)
    solve_sec = time.time() - t1
    if results is None:
        log(f"[{key}] solver FAIL ({solve_sec:.0f}s): {err}")
        return {"key": key, "label": label, "ok": False, "stage": "solver",
                "detail": err, "api_sec": api_sec}

    solved = [r for r in results if r["solved"]]
    turns = [r["turns"] for r in solved]
    hero = next((r for r in results if r["hero"]), None)
    summary = {
        "key": key, "label": label, "ok": True, "api_sec": round(api_sec, 1),
        "solve_sec": round(solve_sec, 1), "code_chars": len(code),
        "n_cubes": len(results), "n_solved": len(solved),
        "solve_rate": round(len(solved) / len(results), 3),
        "hero_solved": bool(hero and hero["solved"]),
        "hero_turns": hero["turns"] if hero and hero["solved"] else None,
        "mean_turns": round(statistics.mean(turns), 2) if turns else None,
        "median_turns": statistics.median(turns) if turns else None,
        "min_turns": min(turns) if turns else None,
        "max_turns": max(turns) if turns else None,
        "results": results,
    }
    log(f"[{key}] done: solved {len(solved)}/{len(results)}, "
        f"mean turns {summary['mean_turns']}, hero {summary['hero_turns']}")
    return summary


def main():
    if not EVAL_CUBES.exists():
        sys.exit(f"missing {EVAL_CUBES} -- run build_eval_set.py first")
    cubes = json.loads(EVAL_CUBES.read_text(encoding="utf-8"))

    sel = set(sys.argv[1:])
    run_models = [m for m in MODELS if not sel or m[0] in sel]
    print(f"Rubik solver compare -- {len(run_models)} model(s), {len(cubes)} cubes")
    print(f"per-cube timeout {PER_CUBE_TIMEOUT:g}s | API timeout {API_TIMEOUT}s | out {OUT_DIR}")
    if sel:
        print(f"Running ONLY: {', '.join(m[0] for m in run_models)}")
    print(flush=True)

    summaries = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(run_models))) as ex:
        futs = [ex.submit(process_model, k, l, f, cubes) for k, l, f in run_models]
        for fut in concurrent.futures.as_completed(futs):
            s = fut.result()
            summaries[s["key"]] = s

    # Merge with any prior results so a SUBSET run preserves the other models
    # (e.g. strict one-shot: a model's first-try result is never re-rolled).
    merged = {}
    sfile = OUT_DIR / "summary.json"
    if sfile.exists():
        try:
            for s in json.loads(sfile.read_text(encoding="utf-8")):
                merged[s["key"]] = s
        except Exception:
            pass
    merged.update(summaries)
    sfile.write_text(
        json.dumps([merged[k] for k, _, _ in MODELS if k in merged], indent=2),
        encoding="utf-8")

    show = [(k, l) for k, l, _ in MODELS if k in merged]
    print(f"\n{'='*72}")
    print(f"{'model':24s} {'solved':>8s} {'rate':>6s} {'hero':>6s} "
          f"{'mean':>6s} {'min':>5s} {'max':>5s}")
    print("-" * 72)
    for key, label in show:
        s = merged.get(key, {})
        if not s.get("ok"):
            print(f"{label:24s}  FAIL @ {s.get('stage','?')}: {str(s.get('detail',''))[:32]}")
            continue
        hero = s["hero_turns"] if s["hero_turns"] is not None else "-"
        print(f"{label:24s} {s['n_solved']:>3d}/{s['n_cubes']:<4d} "
              f"{s['solve_rate']:>6.2f} {str(hero):>6s} "
              f"{str(s['mean_turns'] or '-'):>6s} {str(s['min_turns'] or '-'):>5s} "
              f"{str(s['max_turns'] or '-'):>5s}")
    print(f"{'='*72}")
    print(f"full per-cube detail -> {OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
