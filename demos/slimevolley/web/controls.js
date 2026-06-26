// controls.js: human control of the right (magenta) slime via the keyboard: arrows or WASD to move,
// up / space to jump. createHumanControls() returns { isHuman(), action(), release() }. The player takes
// over on any input and reverts to self-play after ~7s of no input (or immediately on release()).

const IDLE_MS = 7000;

export function createHumanControls() {
  const keys = { left: false, right: false, jump: false };
  let lastInput = -Infinity;

  function onKey(down, e) {
    let hit = true;
    switch (e.code) {
      case 'ArrowLeft': case 'KeyA': keys.left = down; break;
      case 'ArrowRight': case 'KeyD': keys.right = down; break;
      case 'ArrowUp': case 'KeyW': case 'Space': keys.jump = down; break;
      default: hit = false;
    }
    if (hit) { e.preventDefault(); if (down) lastInput = performance.now(); }
  }
  window.addEventListener('keydown', e => onKey(true, e));
  window.addEventListener('keyup', e => onKey(false, e));

  const held = () => keys.left || keys.right || keys.jump;
  return {
    isHuman: () => held() || (performance.now() - lastInput < IDLE_MS),
    action: () => [keys.left ? 1 : 0, keys.right ? 1 : 0, keys.jump ? 1 : 0],
    release: () => { lastInput = -Infinity; keys.left = keys.right = keys.jump = false; },
  };
}
