// app.js: composition root. Loads the roster, creates the match (logic), the renderer (render) and the
// panels + controls (ui), then runs one loop: step the match, draw the court, refresh the panels + HUD.
// All the substance lives in the modules it wires together; this file is just the glue.

import { loadManifest, loadPolicySource, loadExtraInventory } from './manifest.js';
import { createMatch } from './match.js';
import { createRenderer } from './render.js';
import { createPanel } from './panels.js';
import { createHumanControls } from './controls.js';

const TICKS_PER_SEC = 30;
const KIND_BADGE = { code: 'heuristic', rnn: 'baseline' };   // -> the CSS badge colour
const $ = s => document.querySelector(s);

// the curated roster; each code policy carries its source (for the panel + Pyodide)
async function loadRoster() {
  const manifest = await loadManifest();
  const order = [], byId = {};
  for (const p of manifest) {
    const entry = { id: p.id, label: p.label, kind: p.kind, source: null };
    if (p.kind === 'code') entry.source = await loadPolicySource(p);
    byId[p.id] = entry; order.push(p.id);
  }
  for (const e of await loadExtraInventory()) {   // reproduced champions: HTTP-only, optional, appended
    if (!byId[e.id]) { byId[e.id] = e; order.push(e.id); }
  }
  const code = manifest.find(p => p.kind === 'code') || manifest[0];
  const rnn = manifest.find(p => p.kind === 'rnn') || manifest[manifest.length - 1];
  return { order, byId, defaultLeft: code.id, defaultRight: rnn.id };
}

// derive a panel view from a side's live state
function panelView(side, action, isHuman) {
  if (isHuman) return { mode: 'human', buttons: action };
  if (side.kind === 'rnn') return { mode: 'rnn', label: side.label, acts: side.actor.getActs && side.actor.getActs(), buttons: action };
  return { mode: 'code', label: side.label, source: side.source, loading: side.loading, loadError: side.loadError, firedLines: side.actor.firedLines && side.actor.firedLines(), buttons: action };
}

function setHud(which, side, isHuman) {
  const name = $('#' + which + 'Name'), kind = $('#' + which + 'Kind'), score = $('#' + which + 'Score');
  if (isHuman) { name.textContent = 'you'; name.classList.add('isyou'); kind.textContent = 'human'; kind.dataset.kind = ''; score.textContent = ''; return; }
  name.textContent = side.label; name.classList.remove('isyou');
  const badge = KIND_BADGE[side.kind] || ''; kind.textContent = badge || side.kind; kind.dataset.kind = badge;
  score.textContent = '';
}

async function main() {
  const roster = await loadRoster();
  const match = createMatch({ patched: true });
  const renderer = createRenderer($('#cv'));
  const panel = { L: createPanel($('#panelL')), R: createPanel($('#panelR')) };
  const human = createHumanControls();

  const select = (which, id) => { match.selectSide(which, roster.byId[id]); if (which === 'R') human.release(); };
  select('L', roster.defaultLeft);
  select('R', roster.defaultRight);

  // selectors + buttons
  const selL = $('#selL'), selR = $('#selR');
  const fill = (sel, cur) => { sel.innerHTML = roster.order.map(id => `<option value="${id}">${roster.byId[id].label}</option>`).join(''); sel.value = cur; };
  fill(selL, match.sides.L.id); fill(selR, match.sides.R.id);
  const youOpt = document.createElement('option'); youOpt.value = '__you'; youOpt.textContent = '▸ you (playing)'; youOpt.hidden = true; selR.appendChild(youOpt);
  let paused = false, wasHuman = false;
  selL.onchange = () => select('L', selL.value);
  selR.onchange = () => { if (selR.value !== '__you') select('R', selR.value); };
  const togglePause = () => { paused = !paused; $('#pause').textContent = paused ? '▶ play' : '⏸ pause'; };
  $('#pause').onclick = togglePause;
  $('#reset').onclick = () => match.reset();                 // reset the rally only; keep the current driver (ai / you)
  window.addEventListener('keydown', e => {                  // P pause · R reset · Q hand the slime back to its policy
    if (e.target && e.target.tagName === 'SELECT') return;
    if (e.code === 'KeyP') togglePause();
    else if (e.code === 'KeyR') match.reset();
    else if (e.code === 'KeyQ') human.release();
  });

  // the loop: advance the match (logic), draw the court (render), refresh the panels + HUD (ui)
  let acc = 0, prev = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    const humanNow = human.isHuman();
    if (!paused) { acc += dt * TICKS_PER_SEC; let n = Math.floor(acc); acc -= n; while (n-- > 0) match.step(humanNow ? human.action() : null); }

    renderer.draw(match.game, { patched: true, humanRight: humanNow, flash: match.state.flash, flashT: match.state.flashT });
    panel.L.update(panelView(match.sides.L, match.last.L, false));
    panel.R.update(panelView(match.sides.R, match.last.R, humanNow));
    setHud('left', match.sides.L, false);
    setHud('right', match.sides.R, humanNow);
    $('#tally').textContent = `${match.state.scoreL} : ${match.state.scoreR}`;
    const mode = $('#mode'); mode.textContent = humanNow ? '▸ YOU’RE IN · right slime' : '● SELF-PLAY'; mode.className = 'mode ' + (humanNow ? 'on' : '');
    if (humanNow !== wasHuman) { selR.value = humanNow ? '__you' : match.sides.R.id; wasHuman = humanNow; }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main();
