// match.js: the running game. The env, the two sides' policies, the score, and the per-turn logic
// ("how to run a turn"). Pure logic, no DOM, no rendering. createMatch({patched}) returns a controller
// the app drives one turn at a time.

import { Game, C } from './engine.js';
import { POLICY_FACTORY } from './rnn.js';
import { makePyFactory } from './pyrunner.js';

const IDLE = { act: () => [0, 0, 0], reset() {} };      // a policy that does nothing (while Pyodide boots)
const safeAct = (actor, obs) => { try { return actor.act(obs); } catch { return [0, 0, 0]; } };

// one side. The RNN baseline is synchronous JS; a code policy boots Pyodide and swaps in when ready
// (an IDLE placeholder plays meanwhile, so the loop never blocks). `token` drops a stale async load.
function createSide() {
  const side = { id: null, kind: null, label: null, source: null, actor: IDLE, loading: false, loadError: null, token: 0 };
  side.select = entry => {
    const token = ++side.token;
    Object.assign(side, { id: entry.id, kind: entry.kind, label: entry.label, source: entry.source });
    side.loadError = null;
    if (entry.kind !== 'code') { side.actor = POLICY_FACTORY.rnn_baseline(); side.loading = false; return; }
    side.actor = IDLE; side.loading = true;
    makePyFactory(entry.source, entry.label)
      .then(res => { if (side.token === token) { side.actor = (res && res.ok) ? res.factory() : IDLE; side.loadError = (res && res.ok) ? null : ((res && res.reason) || 'load failed'); side.loading = false; } })
      .catch(() => { if (side.token === token) { side.loadError = 'load failed'; side.loading = false; } });
  };
  return side;
}

export function createMatch({ patched }) {
  const game = new Game(Math.random, { patched });
  const sides = { L: createSide(), R: createSide() };
  const last = { L: [0, 0, 0], R: [0, 0, 0] };          // last [fwd,back,jump] per side (for the panels)
  const state = { scoreL: 0, scoreR: 0, flash: null, flashT: 0 };

  function reset() {
    state.scoreL = state.scoreR = 0; state.flash = null; state.flashT = 0;
    game.left.life = game.right.life = C.MAXLIVES; game.reset(true);
    sides.L.actor.reset && sides.L.actor.reset(); sides.R.actor.reset && sides.R.actor.reset();
  }
  function selectSide(which, entry) { sides[which].select(entry); reset(); }

  // Run one turn. rightOverride = [f,b,j] forces the right action (a human is driving); null -> the right
  // policy acts. A rally runs as long as it lasts (no cap, endless rallies are part of the show); a point
  // updates the score; best-of-5-lives shows the win banner.
  function step(rightOverride) {
    if (state.flashT > 0) state.flashT--;
    const aL = safeAct(sides.L.actor, game.obs('left'));
    const aR = rightOverride || safeAct(sides.R.actor, game.obs('right'));
    last.L = aL; last.R = aR;
    const r = game.step(aL, aR);
    if (r > 0) state.scoreR++;
    else if (r < 0) state.scoreL++;
    if (r !== 0 && (game.left.life <= 0 || game.right.life <= 0)) {
      state.flash = game.left.life <= 0 ? (rightOverride ? 'YOU WIN' : 'RIGHT WINS') : 'LEFT WINS';
      state.flashT = 90;
      game.left.life = game.right.life = C.MAXLIVES; game.reset(true);
      sides.L.actor.reset && sides.L.actor.reset(); sides.R.actor.reset && sides.R.actor.reset();
    }
  }

  return { game, sides, last, state, step, reset, selectSide };
}
