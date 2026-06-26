// render.js: draws the court to a canvas: gradient field, coordinate grid, chalk floor, the net, the ball
// (with a comet trail) and the two slimes, plus the env's true next-landing crosshair. Pure presentation:
// it reads game state and never mutates it.

import { C } from './engine.js';

const SCALE = 15;     // px per world unit
const VIEW_H = 20;    // world units shown vertically
const W = C.REF_W * SCALE;
const H = VIEW_H * SCALE;

const PALETTE = {
  courtTop: '#f4f8fb', courtBot: '#e7eef4', vignette: 'rgba(70,95,120,0.08)',
  grid: 'rgba(64,108,150,0.11)', gridMid: 'rgba(64,108,150,0.24)',
  floor: '#dbe5ee', chalk: '#5f7384', shadow: 'rgba(40,60,80,0.14)',
  mint: '#12a98e', mintDark: '#0b7d68', magenta: '#e23d7b', magentaDark: '#b32a60',
  ball: '#33454f', ballDark: '#1d2a32', ballHi: '#8597a2',
  trail: '70,92,108', target: '#8854d0', eye: '#17242c', text: '#2a3a45',
};

export function createRenderer(canvas) {
  const ctx = canvas.getContext('2d');
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const reducedMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const TRAIL_MAX = reducedMotion ? 4 : 16;
  const trail = [];
  let frame = 0;

  const wx = x => (x + C.REF_W / 2) * SCALE;   // world x -> canvas px
  const wy = y => H - y * SCALE;               // world y (up) -> canvas px

  function fit() {
    const cssW = canvas.clientWidth, cssH = cssW * (H / W);
    canvas.style.height = cssH + 'px';
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  fit();
  window.addEventListener('resize', fit);

  // Forward-integrate the live ball (env physics, wall + net bounces, slimes ignored) to its floor contact.
  function predictLanding(game, patched) {
    if (game.delay > 0) return null;
    let x = game.ball.x, y = game.ball.y, vx = game.ball.vx, vy = game.ball.vy, px = x;
    const r = game.ball.r, F = C.FRICTION, N = C.NUDGE * C.DT, mx = C.MAX_BALL;
    for (let i = 0; i < 160; i++) {
      vy += C.GRAVITY * C.DT;
      const m2 = vx * vx + vy * vy;
      if (m2 > mx * mx) { const m = Math.sqrt(m2); vx = vx / m * mx; vy = vy / m * mx; }
      px = x; x += vx * C.DT; y += vy * C.DT;
      if (x <= (r - C.REF_W / 2)) { vx *= -F; x = r - C.REF_W / 2 + N; }
      if (x >= (C.REF_W / 2 - r)) { vx *= -F; x = C.REF_W / 2 - r - N; }
      if (patched && y <= C.WALL_H) {
        const NETX = C.WALL_W / 2, cx = Math.max(-NETX, Math.min(x, NETX)), cy = Math.max(C.REF_U, Math.min(y, C.WALL_H));
        const ex = x - cx, ey = y - cy, e2 = ex * ex + ey * ey;
        if (e2 < r * r) {
          if (e2 < 1e-12) { const s = px >= 0 ? 1 : -1; x = s * (NETX + r) + s * N; vx = s * Math.abs(vx) * F; if (vy < 0) vy = -vy * F; }
          else { const d = Math.sqrt(e2), nx = ex / d, ny = ey / d; x = cx + nx * r; y = cy + ny * r; const vn = vx * nx + vy * ny; vx = (vx - 2 * vn * nx) * F; vy = (vy - 2 * vn * ny) * F; }
        }
      }
      if (y <= (r + C.REF_U)) return x;
      if (y >= (C.REF_H - r)) { vy *= -F; y = C.REF_H - r - N; }
    }
    return null;
  }

  function drawSlime(game, agent, color, colorDark, dir, glow) {
    const cx = wx(agent.x), cy = wy(agent.y), r = agent.r * SCALE;
    if (glow) { ctx.strokeStyle = 'rgba(226,61,123,0.9)'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, r + 4, Math.PI, 0); ctx.stroke(); }
    ctx.fillStyle = PALETTE.shadow;
    ctx.beginPath(); ctx.ellipse(cx, wy(C.REF_U) + 2, r * 0.95, r * 0.32, 0, 0, Math.PI * 2); ctx.fill();
    const grad = ctx.createLinearGradient(cx, cy - r, cx, cy);
    grad.addColorStop(0, color); grad.addColorStop(1, colorDark);
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0); ctx.closePath(); ctx.fill();
    const ex = cx + dir * r * 0.42, ey = cy - r * 0.45;
    const dx = wx(game.ball.x) - ex, dy = wy(game.ball.y) - ey, dd = Math.hypot(dx, dy) || 1;
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(ex, ey, r * 0.22, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = PALETTE.eye; ctx.beginPath();
    ctx.arc(ex + dx / dd * r * 0.09, ey + dy / dd * r * 0.09, r * 0.11, 0, Math.PI * 2); ctx.fill();
  }

  function draw(game, { patched, humanRight = false, flash = null, flashT = 0 }) {
    frame++;
    ctx.save(); ctx.scale(canvas.width / W, canvas.height / H);
    const floorY = wy(C.REF_U);

    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, PALETTE.courtTop); bg.addColorStop(1, PALETTE.courtBot);
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

    ctx.lineWidth = 1;
    for (let gx = -C.REF_W / 2; gx <= C.REF_W / 2 + 0.01; gx += 4) {
      const X = wx(gx); ctx.strokeStyle = Math.abs(gx) < 0.01 ? PALETTE.gridMid : PALETTE.grid;
      ctx.beginPath(); ctx.moveTo(X, 0); ctx.lineTo(X, floorY); ctx.stroke();
    }
    for (let gy = 4; gy <= VIEW_H; gy += 4) {
      const Y = wy(gy); ctx.strokeStyle = PALETTE.grid;
      ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(W, Y); ctx.stroke();
    }

    ctx.fillStyle = PALETTE.floor; ctx.fillRect(0, floorY, W, H - floorY);
    ctx.strokeStyle = PALETTE.chalk; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(0, floorY); ctx.lineTo(W, floorY); ctx.stroke();

    const land = predictLanding(game, patched);
    if (land != null) {
      const lx = wx(land), pulse = reducedMotion ? 0 : (Math.sin(frame * 0.16) * 0.5 + 0.5);
      ctx.save(); ctx.strokeStyle = PALETTE.target;
      ctx.globalAlpha = 0.18; ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
      ctx.beginPath(); ctx.moveTo(lx, floorY - 34); ctx.lineTo(lx, floorY); ctx.stroke();
      ctx.globalAlpha = 0.65 + 0.3 * pulse; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.arc(lx, floorY, 10 + 2 * pulse, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]); ctx.beginPath();
      ctx.moveTo(lx - 15, floorY); ctx.lineTo(lx + 15, floorY);
      ctx.moveTo(lx, floorY - 15); ctx.lineTo(lx, floorY - 3); ctx.stroke();
      ctx.globalAlpha = 0.9; ctx.fillStyle = PALETTE.target;
      ctx.beginPath(); ctx.arc(lx, floorY, 2.6, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }

    const nx = wx(0);
    ctx.fillStyle = '#aab8c4';
    ctx.fillRect(nx - C.WALL_W / 2 * SCALE, wy(C.WALL_H), C.WALL_W * SCALE, floorY - wy(C.WALL_H));
    ctx.fillStyle = PALETTE.chalk;
    ctx.beginPath(); ctx.arc(nx, wy(C.WALL_H), C.STUB_R * SCALE, 0, Math.PI * 2); ctx.fill();

    if (game.delay > 0) trail.length = 0;
    else { trail.push({ x: game.ball.x, y: game.ball.y }); if (trail.length > TRAIL_MAX) trail.shift(); }
    ctx.lineCap = 'round';
    for (let i = 0; i < trail.length - 1; i++) {
      const a = (i + 1) / trail.length;
      ctx.strokeStyle = `rgba(${PALETTE.trail},${(a * 0.45).toFixed(3)})`;
      ctx.lineWidth = a * game.ball.r * SCALE * 1.05;
      ctx.beginPath(); ctx.moveTo(wx(trail[i].x), wy(trail[i].y)); ctx.lineTo(wx(trail[i + 1].x), wy(trail[i + 1].y)); ctx.stroke();
    }

    const bx = wx(game.ball.x), by = wy(game.ball.y), br = game.ball.r * SCALE;
    const ballGrad = ctx.createRadialGradient(bx - br * 0.3, by - br * 0.3, br * 0.15, bx, by, br);
    ballGrad.addColorStop(0, PALETTE.ballHi); ballGrad.addColorStop(0.55, PALETTE.ball); ballGrad.addColorStop(1, PALETTE.ballDark);
    ctx.fillStyle = ballGrad; ctx.beginPath(); ctx.arc(bx, by, br, 0, Math.PI * 2); ctx.fill();

    drawSlime(game, game.left, PALETTE.mint, PALETTE.mintDark, 1, false);
    drawSlime(game, game.right, PALETTE.magenta, PALETTE.magentaDark, -1, humanRight);

    ctx.fillStyle = PALETTE.mint;
    for (let i = 0; i < game.left.life; i++) { ctx.beginPath(); ctx.arc(18 + i * 14, 18, 4, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = PALETTE.magenta;
    for (let i = 0; i < game.right.life; i++) { ctx.beginPath(); ctx.arc(W - 18 - i * 14, 18, 4, 0, Math.PI * 2); ctx.fill(); }

    const vg = ctx.createRadialGradient(W / 2, H * 0.42, H * 0.32, W / 2, H * 0.5, H * 0.98);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, PALETTE.vignette);
    ctx.fillStyle = vg; ctx.fillRect(0, 0, W, H);

    if (flashT > 0 && flash) {
      ctx.globalAlpha = Math.min(1, flashT / 30);
      const tw = 340, th = 50;
      ctx.fillStyle = 'rgba(255,255,255,0.93)'; ctx.fillRect(W / 2 - tw / 2, H / 2 - th / 2, tw, th);
      ctx.strokeStyle = 'rgba(18,169,142,0.6)'; ctx.lineWidth = 1; ctx.strokeRect(W / 2 - tw / 2, H / 2 - th / 2, tw, th);
      ctx.fillStyle = PALETTE.text; ctx.font = '600 18px ui-monospace,Menlo,monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(flash, W / 2, H / 2);
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  return { draw };
}
