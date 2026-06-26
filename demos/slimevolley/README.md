# Slime Volley: a learned-policy playground

Agentically-evolved, interpretable control policies, each written as code by a different model, playing
solid-net Slime Volley on this webpage. Pick any two, watch them rally, and see which lines of each policy's
logic fire as it plays. Or jump in and play yourself.

## Run it
Open `slimevolley.html` directly (double-click or `file://`); no server, no build.

Or serve it locally:

```
python3 -m http.server
```

then open `http://localhost:8000/slimevolley.html`.

## The policies and what you can do
Choose from models that produced policies. **Fugu**, **Gemini**, and **Claude** are each evolved as code by a different agentic model
and run as real, self-contained Python via Pyodide, alongside the built-in **RNN baseline** opponent. Their
source is in `policies/`.  Pick a policy on each side. Code policies show their
real Python with the executing line highlighted; the RNN shows its observation → recurrent state → logits.
Arrow keys / WASD take the magenta slime (↑ / Space to jump, **Q** to hand it back); **P** pauses, **R** resets.

## Retrain the policies
Every policy here was produced by an agentic LLM training loop (the **Fugu (another run)** entry is a fresh re-run
of fugu). To re-run that loop and train new champions yourself, see [`train/`](train/); it outputs a drop-in
`make_policy` you can add to `policies/`.

## Refine website
The page is built from the source in `web/` (which ships here too). Edit `web/`, or add a policy: drop a
`policies/<id>.py` exposing `make_policy(seat=0) → act(obs) → [forward, backward, jump]`, add a row to
`policies.json`, then rebuild with `node web/build.mjs`.

## Credits
Agentic policy-optimization framework and frontend by Yingtao Tian, with help of Claude Code. Physics and RNN
baseline from [slimevolleygym](https://github.com/hardmaru/slimevolleygym) (David Ha).
