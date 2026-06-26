# Write a Slime Volley policy that beats the baseline

You are evolving one control policy for solid-net Slime Volley, iteration by iteration, until it WINS games against
the built-in baseline. You win a game by landing the ball in the opponent's open court. Your job each iteration:
make the policy win more games than it did last time, and leave it where the grader can find it. This file gives you
the env contract, the loop you run, and a wide range of strategy you are expected to use. Read it in full: the
difference between a 50% drawer that just survives and a real winner is the strategy below.

## The three nested loops

You operate at three nested levels. Hold all three in mind at once.

1. **The control loop (inside one game).** `act(obs)` is called every single step with a fresh observation. Use it
   as FEEDBACK, not as a script. Sense the ball, your slime, and the opponent each step and correct. A policy that
   emits a pre-planned (open-loop) action sequence and barely reads `obs` is brittle: it cannot correct drift, a bad
   bounce, or the opponent's response, and it breaks the instant the world deviates from your assumption.
2. **The improvement loop (one iteration).** EVAL the current policy, READ a game to see WHY, make ONE deliberate
   EDIT, re-score, and KEEP it only if the held-out win-rate did not regress. You run this loop once per iteration;
   do it well.
3. **The search loop (across iterations).** Keep an archive/pool of candidates; choose your search method; when
   local tweaks stop helping, change a mechanism or restart from a different pool member. Do not only carry the
   latest champion. This is where you escape a plateau.

## The improvement loop, in detail (EVAL -> READ -> EDIT -> KEEP)

**EVAL.** Score the current policy with `uv run python assess.py --file kit/policies/champion.py`. It plays 500
games vs the baseline and prints `score = (wins + 0.5*draws) / games`. This is the same scorer that grades you, so
trust it over any number you compute yourself. It is your held-out check; for fast inner-loop selection you may also
build a denser signal (see below).

**READ a single game in detail.** Write a TEXT rollout of one representative game: one line per step with your slime
x/y/vx/vy, the opponent's, the ball x/y/vx/vy, your action, the reward, and notable events (point conceded, point
won, jump fired). Then read that game closely; never render video or images, the text is enough and faster. From
the one game, do four things:

- **LOOK.** Where does every point come from, yours and the opponent's? At each moment you CONCEDE, pin the cause:
  late to the ball, mistimed jump, wrong interception height, out of position, hit it into your own court. At each
  moment you SCORE, what set it up? An edit that amplifies a working attack is worth as much as one that patches a
  hole.
- **MODEL.** Build an explicit (even rough) model of the dynamics from the trace: what each of your actions does to
  your slime; where the ball goes next given its position and velocity; when it will cross the net; where it will
  land. Verify the model against the trace on a few key points. A wrong model steers every later edit wrong.
- **THINK about how to WIN.** Brainstorm concrete strategies that could help: specific offensive setups, where and
  when to attack, moving to the predicted landing spot early, spiking the ball down from height, varying your
  returns so the baseline cannot settle. Identify which of the OPPONENT'S MISTAKES you can exploit: where it stands,
  when it jumps, what kinds of balls it mishandles. The only wins ever seen here came from reproducing situations
  the baseline misplays, so hunt for those. And think LONG-TERM: what overall strategy are you building toward
  across iterations, and which single situation, if you solved it, would convert the most draws into wins?
  Decompose the game into named situations (e.g. "deep ball to my back court", "high ball at the net", "fast flat
  return") and the winning response to each.
- **EXPLAIN.** Before you change anything, WRITE DOWN where the current policy wins, draws, and loses, and WHY in
  each case. Make the trade-offs explicit on the page; that written analysis is what should drive the single change
  you make. One real measurement is worth ten speculative edits.

**EDIT.** Start from your BEST policy so far and make ONE deliberate change, guided by what you read. Do not rewrite
from scratch: a fresh rewrite almost always scores below the peak you already carried, and you lose what worked.

**KEEP or REVERT.** Re-score. Keep the change only if the win-rate did not regress; otherwise revert and try a
different idea. Keep a candidate that wins differently even if it scores a touch lower (see the pool, below).

## Play to win, not to survive

This is the single most important mindset, and ignoring it is the most common failure here.

- Winning is not the same as not-losing. Surviving, stalling, or drawing is NOT the objective; WINNING is. A policy
  that avoids losing but rarely scores has FAILED the goal. A run that settles into mostly-draws and never climbs is
  the classic trap.
- Work out exactly how a point is scored, then act deliberately to PRODUCE that outcome.
- Treat a large DRAW bucket as UNCONVERTED WINS: it is your headroom, not a safe place to settle. Convert draws.
- Think and act over a LONGER horizon than the next reaction: build sequences that SET UP and then CONVERT a winning
  opportunity. Reactive one-step play caps low; be willing to commit and be aggressive when a line leads to a win.
- Track and report explicit WIN / DRAW / LOSS counts, not just a mean reward or margin (a margin that improves but
  stays negative is still losing).

## Escape plateaus: change the mechanism, restart from the pool

When you stop improving, do not keep tuning the same constants.

- A constant is not a strategy. If several local tweaks in a row do not move the held-out win-rate, STOP tuning and
  make a STRUCTURAL change: decompose your losses into named situations and change a MECHANISM (how you decide, not
  just a threshold).
- Restart or branch from a DIFFERENT member of your pool, not only the latest champion. Novelty,
  restart-from-archive, and hall-of-fame are all valid even against a fixed opponent; reach for them when stuck in a
  local optimum.
- Never let one bad early experience permanently close off a whole class of action. If you concluded "never jump" or
  "never advance" or "a fixed sequence is best", RE-TEST it later under a sharper condition; a high-variance early
  failure is not a proof.

## Choose your own means (agency)

How you optimize is entirely yours. Probe the env as a black box; write your own evaluators, probes, and
visualizers; and use ANY search method (hand-tuning, random/grid search, evolutionary/CEM, MAP-Elites, bandits,
restart-from-archive). You may install any package you need FOR YOUR OWN SEARCH AND ANALYSIS scripts. (The champion
you submit stays self-contained, see the policy rules below; this freedom is for your tooling, not the shipped
policy.) Nothing here prescribes a tactic; the score decides what works.

## Model, predict, and close the loop

- Build a model and act on PREDICTION. From the traces, form an explicit model of where the ball goes next, and act
  on the prediction, not just the current frame. Move to where the ball WILL be.
- Plan within the episode where you can. Look a few steps ahead of THIS game with your model (a short rollout /
  lookahead / MPC, distinct from a search trial) and pick the action with the best predicted outcome, rather than a
  pure reflex constant.
- Close the control loop. `act(obs)` gets a fresh observation every step; sense the error each step and correct it.
  Never ship an open-loop scripted sequence.
- Model the opponent. The baseline is fixed, so it has HABITS to exploit; anticipate its move and aim at the best
  response to it.

## Design and validate a dense selection signal

Win-rate vs the full-strength baseline is SPARSE while you are weak: it gives almost no gradient, so most edits look
equal. The obvious proxies are TRAPS.

- RECORDED TRAP: rally-length, survival-time, and point-rate signals all peak at LONG LOSSES, and every lineage that
  climbed them finished at 0% wins. Your selection signal must move when you get CLOSER to TAKING points, not when
  you merely survive longer.
- Design a denser signal and validate it: (a) state why its MAXIMUM coincides with winning, not with surviving
  longer or losing by less; (b) check it against reality as you go, and when the held-out win-rate improves your
  signal should have moved first.
- ALARM: if your signal has climbed for several consecutive trials while the true win-rate has NOT moved, it is
  misaligned. STOP climbing it; redesigning the measurement IS your next edit.
- You may construct intermediate opponents for SELECTION ONLY to restore a gradient while weak, e.g. a handicapped
  copy of the baseline (action delay or noise) or snapshots from your own archive. The HELD-OUT stays the
  full-strength baseline win-rate and is never selected on.

## Keep, grow, and diversify a pool of candidates

Your retained candidates are your memory; treat the pool as a first-class object.

- Keep an ARCHIVE/POOL of behaviorally-DISTINCT candidates, not just one champion. The champion you submit is your
  current best PICK from the pool, never the only thing you keep.
- ITERATIVELY ADD to it: whenever you find a candidate that wins in a meaningfully DIFFERENT way, keep it even if
  another scores a touch higher. Never discard a different-winning candidate; keep it as a reference and a hedge.
- DIVERSITY is yours to define. Discover the behavioral AXES that matter for THIS env (offense vs defense, tempo,
  risk appetite, positioning, where and when it attacks, when-to-commit); there is no fixed list. Differentiate
  candidates by their CONCRETE behavior, their actual WIN/DRAW/LOSS profile and what they physically DO, not by
  their label. Keep candidates that genuinely SPAN your axes; prune near-duplicates so the pool stays informative.
- Reason over BOTH horizons: the SHORT, the single best next change to one candidate; and the LONG, which niche, or
  which RECOMBINATION across niches (e.g. one candidate's offense grafted onto another's robustness), wins most
  reliably over time. When stuck, restart or branch from any pool member.

## The environment

The env is a local, vendored build of `slimevolleygym` with a SOLID net: a ball driven into the net rebounds to the
side that hit it (it does not slip through and score at the base). It predates NumPy 2, so add two shims before
importing it:

```python
import sys
import numpy as np
for name, value in {"bool8": np.bool_, "float_": np.float64}.items():
    if not hasattr(np, name):
        setattr(np, name, value)
sys.path.insert(0, "env")
from slimevolleygym.slimevolley import SlimeVolleyEnv

env = SlimeVolleyEnv()
obs = env.reset()                 # 12-d float vector
obs, reward, done, info = env.step([forward, backward, jump])
```

This import is for probing the env in your own throwaway scripts. Your `champion.py` must NOT import it (see
"Self-contained policy" below).

- **Observation:** 12 numbers, scaled by 1/10, from your slime's own side: `[me x, me y, me vx, me vy, ball x,
  ball y, ball vx, ball vy, opp x, opp y, opp vx, opp vy]`. Multiply by 10 for real units.
- **Action:** three buttons `[forward, backward, jump]`, each 0 or 1.
- **Opponent:** the built-in baseline plays the other side. You play AGAINST it; study its behavior in the traces
  and exploit its mistakes, but do not import or copy its policy. You may also play archived snapshots of your own
  policies for self-play.
- **Reward:** +1 when you score, -1 when you concede, 0 otherwise. A game ends when one side wins the point or the
  rally hits the step cap (a tie at the cap counts as a draw).

## The policy contract (this is what gets graded)

```python
def make_policy(seat=0):
    def act(obs):
        # obs: the 12-d vector above. Return three buttons.
        return [forward, backward, jump]   # each 0 or 1
    return act
```

## Self-contained policy (REQUIRED)

Your `champion.py` ships to a browser and runs there via Pyodide, so it must stand entirely on its own:

- **Use only `numpy` and the Python standard library.** No other imports. In particular, do NOT `import
  slimevolleygym` or the env inside `champion.py` (the env import above is for your own throwaway probing scripts).
- **Do not copy the opponent.** Do not import or copy the baseline's (or any opponent's) policy into your champion,
  for example `BaselinePolicy`: that is not your own evolved logic, and it would not run in the browser. You SHOULD
  study the opponent's BEHAVIOR in the rollout traces and exploit its mistakes; you must not import its code. If you
  need to anticipate the physics to plan ahead, inline the small amount you need yourself.
- On a new best, SIMPLIFY to the shortest code that holds the gain.

A champion that imports the env or the opponent's policy is rejected as not self-contained.

## Instrument first

Two cheap steps before you optimize pay for themselves: probe the obs/action semantics with a controlled probe (one
action at a time, watch what moves in the trace) so your model is right; and measure a failure before you fix it.

## What to leave behind (exactly these)

- `kit/policies/champion.py` : your best policy, defining `make_policy(seat=0)` as above. You may keep extra
  candidate policies in `kit/policies/`; the grader enumerates and ranks all of them, so leave your whole pool.
- `note.md` : a RICH note for the next iteration (the memory handed forward; make it substantial, not a thin
  summary). Include: the important findings and discussions; the key traits / parameters that matter and their
  validated values; the current diversity POOL (which candidates you keep, the behavioral axes you chose, how each
  one wins differently); what you LEARNED this iteration (what works, what does not, the dead ends you ruled out);
  and what to try next. No policy code in the note.

Do everything inside this directory. The env package and `assess.py` are already here. When you are done, leave the
deliverables above (plus your pool of candidate policies) and stop.
