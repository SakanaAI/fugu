// engine.js: faithful JS port of slime-volley physics (slimevolleygym, David Ha, 2020), solid-net variant.
// Units: x in [-24,24], y up from 0, ground top at y=1.5. One physics step == 1/30 s.

export const C = {
  REF_W: 48, REF_H: 48,
  REF_U: 1.5,                 // ground height (top of ground)
  WALL_W: 1.0, WALL_H: 3.5,   // fence
  SPEED_X: 10 * 1.75,         // 17.5
  SPEED_Y: 10 * 1.35,         // 13.5
  MAX_BALL: 15 * 1.5,         // 22.5
  DT: 1 / 30,
  NUDGE: 0.1,
  FRICTION: 1.0,
  DELAY_FRAMES: 30,
  GRAVITY: -9.8 * 2 * 1.5,    // -29.4
  MAXLIVES: 5,
  BALL_R: 0.5, AGENT_R: 1.5, STUB_R: 0.5,
};

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

export class Particle {
  constructor(x, y, vx, vy, r) {
    this.x = x; this.y = y; this.prevx = x; this.prevy = y;
    this.vx = vx; this.vy = vy; this.r = r;
  }
  move() { this.prevx = this.x; this.prevy = this.y; this.x += this.vx * C.DT; this.y += this.vy * C.DT; }
  applyAccel(ax, ay) { this.vx += ax * C.DT; this.vy += ay * C.DT; }
  limitSpeed(mn, mx) {
    const m2 = this.vx * this.vx + this.vy * this.vy;
    if (m2 > mx * mx) { const m = Math.sqrt(m2); this.vx = this.vx / m * mx; this.vy = this.vy / m * mx; }
    if (m2 < mn * mn && m2 > 0) { const m = Math.sqrt(m2); this.vx = this.vx / m * mn; this.vy = this.vy / m * mn; }
  }
  checkEdges(patched) {                       // patched=true -> SOLID net (the ball rebounds off the net bar)
    const r = this.r, F = C.FRICTION, N = C.NUDGE * C.DT;
    if (this.x <= (r - C.REF_W / 2)) { this.vx *= -F; this.x = r - C.REF_W / 2 + N; }
    if (this.x >= (C.REF_W / 2 - r)) { this.vx *= -F; this.x = C.REF_W / 2 - r - N; }
    // PATCHED solid net: resolve ball vs the net BAR (rounded-rect) BEFORE scoring the floor, so a ball
    // driven into the net rebounds to the hitter instead of slipping through to the base.
    if (patched && this.y <= C.WALL_H) {
      const NETX = C.WALL_W / 2;
      const cx = Math.max(-NETX, Math.min(this.x, NETX));
      const cy = Math.max(C.REF_U, Math.min(this.y, C.WALL_H));
      const dx = this.x - cx, dy = this.y - cy, d2 = dx * dx + dy * dy;
      if (d2 < r * r) {
        if (d2 < 1e-12) {                     // center inside the bar (jam): pop to the hitter's side + lift
          const side = this.prevx >= 0 ? 1 : -1;
          this.x = side * (NETX + r) + side * N;
          this.vx = side * Math.abs(this.vx) * F;
          if (this.vy < 0) this.vy = -this.vy * F;
        } else {
          const d = Math.sqrt(d2), nx = dx / d, ny = dy / d;
          this.x = cx + nx * r; this.y = cy + ny * r;
          const vn = this.vx * nx + this.vy * ny;
          this.vx = (this.vx - 2 * vn * nx) * F;
          this.vy = (this.vy - 2 * vn * ny) * F;
        }
      }
    }
    if (this.y <= (r + C.REF_U)) {
      this.vy *= -F; this.y = r + C.REF_U + N;
      return this.x <= 0 ? -1 : 1;            // ground hit -> which side scored on
    }
    if (this.y >= (C.REF_H - r)) { this.vy *= -F; this.y = C.REF_H - r - N; }
    // fence (original crossing-only reflection, only when NOT patched)
    if (!patched && this.x <= (C.WALL_W / 2 + r) && this.prevx > (C.WALL_W / 2 + r) && this.y <= C.WALL_H) {
      this.vx *= -F; this.x = C.WALL_W / 2 + r + N;
    }
    if (!patched && this.x >= (-C.WALL_W / 2 - r) && this.prevx < (-C.WALL_W / 2 - r) && this.y <= C.WALL_H) {
      this.vx *= -F; this.x = -C.WALL_W / 2 - r - N;
    }
    return 0;
  }
  dist2(p) { const dx = p.x - this.x, dy = p.y - this.y; return dx * dx + dy * dy; }
  isColliding(p) { const r = this.r + p.r; return r * r > this.dist2(p); }
  bounce(p) {
    let abx = this.x - p.x, aby = this.y - p.y;
    const abd = Math.sqrt(abx * abx + aby * aby) || 1e-9;
    abx /= abd; aby /= abd;
    const nx = abx, ny = aby;
    abx *= C.NUDGE; aby *= C.NUDGE;
    let guard = 0;
    while (this.isColliding(p) && guard++ < 100) { this.x += abx; this.y += aby; }
    let ux = this.vx - p.vx, uy = this.vy - p.vy;
    const un = ux * nx + uy * ny;
    ux -= nx * un * 2; uy -= ny * un * 2;
    this.vx = ux + p.vx; this.vy = uy + p.vy;
  }
}

export class Agent {
  constructor(dir, x) { this.dir = dir; this.x = x; this.y = 1.5; this.r = C.AGENT_R; this.vx = 0; this.vy = 0; this.dvx = 0; this.dvy = 0; this.life = C.MAXLIVES; }
  setAction(a) {
    const fwd = a[0] > 0, back = a[1] > 0, jmp = a[2] > 0;
    this.dvx = 0; this.dvy = 0;
    if (fwd && !back) this.dvx = -C.SPEED_X;
    if (back && !fwd) this.dvx = C.SPEED_X;
    if (jmp) this.dvy = C.SPEED_Y;
  }
  update() {
    this.vy += C.GRAVITY * C.DT;
    if (this.y <= C.REF_U + C.NUDGE * C.DT) this.vy = this.dvy;
    this.vx = this.dvx * this.dir;
    this.x += this.vx * C.DT; this.y += this.vy * C.DT;
    if (this.y <= C.REF_U) { this.y = C.REF_U; this.vy = 0; }
    if (this.x * this.dir <= (C.WALL_W / 2 + this.r)) { this.vx = 0; this.x = this.dir * (C.WALL_W / 2 + this.r); }
    if (this.x * this.dir >= (C.REF_W / 2 - this.r)) { this.vx = 0; this.x = this.dir * (C.REF_W / 2 - this.r); }
  }
}

export class Game {
  constructor(rng, opts) { this.rng = rng || Math.random; this.patched = !!(opts && opts.patched); this.reset(true); }
  _ball() {
    const vx = (this.rng() * 40 - 20);          // uniform[-20,20]
    const vy = (this.rng() * 15 + 10);          // uniform[10,25]
    return new Particle(0, C.REF_W / 4, vx, vy, C.BALL_R);
  }
  reset(full) {
    this.ball = this._ball();
    this.fenceStub = new Particle(0, C.WALL_H, 0, 0, C.STUB_R);
    if (full || !this.left) {
      this.left = new Agent(-1, -C.REF_W / 4);
      this.right = new Agent(1, C.REF_W / 4);
    } else {
      this.left.x = -C.REF_W / 4; this.left.y = 1.5; this.left.vx = this.left.vy = 0;
      this.right.x = C.REF_W / 4; this.right.y = 1.5; this.right.vx = this.right.vy = 0;
    }
    this.delay = C.DELAY_FRAMES;
  }
  newMatch() { this.ball = this._ball(); this.delay = C.DELAY_FRAMES; }
  // one step. actionLeft/actionRight = [fwd,back,jump]. returns +1 if right scored, -1 if left scored, 0 else
  step(actionLeft, actionRight) {
    this.left.setAction(actionLeft); this.right.setAction(actionRight);
    this.left.update(); this.right.update();
    // delayScreen.status(): frozen while counting down, ball moves once delay hits 0
    let ballLive = false;
    if (this.delay === 0) ballLive = true; else this.delay -= 1;
    if (ballLive) { this.ball.applyAccel(0, C.GRAVITY); this.ball.limitSpeed(0, C.MAX_BALL); this.ball.move(); }
    if (this.ball.isColliding(this.left)) this.ball.bounce(this.left);
    if (this.ball.isColliding(this.right)) this.ball.bounce(this.right);
    if (this.ball.isColliding(this.fenceStub)) this.ball.bounce(this.fenceStub);
    const result = -this.ball.checkEdges(this.patched);     // +1 right scores, -1 left scores
    if (result !== 0) {
      this.newMatch();
      if (result < 0) this.right.life -= 1; else this.left.life -= 1;
    }
    return result;
  }
  // observation from one side's perspective (dir-normalized), scaled /10, matches getObservation()
  obs(side) {
    const me = side === 'right' ? this.right : this.left;
    const op = side === 'right' ? this.left : this.right;
    const d = me.dir, b = this.ball;
    const o = [
      me.x * d, me.y, me.vx * d, me.vy,
      b.x * d, b.y, b.vx * d, b.vy,
      op.x * (-d), op.y, op.vx * (-d), op.vy,
    ];
    return o.map(v => v / 10);
  }
}
