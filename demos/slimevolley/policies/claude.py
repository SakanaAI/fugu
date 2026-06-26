import numpy as np
for _k, _v in {"bool8": np.bool_, "float_": np.float64}.items():
    if not hasattr(np, _k):
        setattr(np, _k, _v)

# raw slime-volley env units (shared by the strategy below + the physics at the bottom)
TIMESTEP = 1 / 30.0
GRAVITY = -29.4
MAX_BALL_SPEED = 22.5
REF_U = 1.5
REF_W = 48.0
REF_WALL_WIDTH = 1.0
REF_WALL_HEIGHT = 3.5
PLAYER_SPEED_X = 17.5
PLAYER_SPEED_Y = 13.5
NUDGE = 0.1
DT, GRAV, MAXB, NET_X = TIMESTEP, GRAVITY, MAX_BALL_SPEED, 0.5

PARAMS = dict(
    contact_h=2.7, hit_offset=1.4, deadzone=0.4,
    mpc_h=7.0, mpc_dx=9.0, attack_floor=0.0,
    depth_w=0.5, apex_w=1.5, apex_ok=7.0, home_x=6.0,
)

DRIVE, STAND, BACK = [1, 0, 0], [0, 0, 0], [0, 1, 0]
_LIB = {
    "drive": [DRIVE] * 30,
    "jumpdrive": [[1, 0, 1]] + [DRIVE] * 30,
    "wait1drive": [STAND] * 1 + [DRIVE] * 30,
    "wait2drive": [STAND] * 2 + [DRIVE] * 30,
    "back1drive": [BACK] * 1 + [DRIVE] * 30,
    "back2drive": [BACK] * 2 + [DRIVE] * 30,
    "stand": [STAND] * 30,
    "jumpstand": [[0, 0, 1]] + [STAND] * 30,
    "back1stand": [BACK] * 1 + [STAND] * 30,
}
WALL = REF_W / 2 - 0.5


# Predict the ball's contact point on my side by forward-simulating its flight (solid-net rebounds included).
def _predict_contact(bx, by, bvx, bvy, contact_h):
    ball = Particle(bx, by, bvx, bvy, 0.5, c=(0, 0, 0))
    fs = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2, c=(0, 0, 0))
    for i in range(200):
        ball.applyAcceleration(0, GRAV); ball.limitSpeed(0, MAXB); ball.move()
        if ball.isColliding(fs):
            ball.bounce(fs)
        if ball.checkEdges() != 0:
            return (ball.x, i)
        if ball.vy < 0 and ball.y <= contact_h and ball.x > NET_X:
            return (ball.x, i)
    return (None, None)


def _agent_update(ag, action):
    fwd, bwd, jmp = action[0] > 0, action[1] > 0, action[2] > 0
    dvx = -PLAYER_SPEED_X if (fwd and not bwd) else (PLAYER_SPEED_X if (bwd and not fwd) else 0.0)
    dvy = PLAYER_SPEED_Y if jmp else 0.0
    ag.vy += GRAV * DT
    if ag.y <= REF_U + NUDGE * DT:
        ag.vy = dvy
    ag.vx = dvx * ag.dir
    ag.x += ag.vx * DT; ag.y += ag.vy * DT
    if ag.y <= REF_U:
        ag.y = REF_U; ag.vy = 0
    if ag.x * ag.dir <= (REF_WALL_WIDTH / 2 + ag.r):
        ag.vx = 0; ag.x = ag.dir * (REF_WALL_WIDTH / 2 + ag.r)
    if ag.x * ag.dir >= (REF_W / 2 - ag.r):
        ag.vx = 0; ag.x = ag.dir * (REF_W / 2 - ag.r)


# Short receding-horizon lookahead (MPC): simulate me + the ball under each timed action sequence.
def _rollout(bx, by, bvx, bvy, sx, sy, svx, svy, seq):
    ball = Particle(bx, by, bvx, bvy, 0.5, c=(0, 0, 0))
    ag = Agent(1, sx, sy, c=(0, 0, 0)); ag.vx, ag.vy = svx, svy
    fs = Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2, c=(0, 0, 0))
    hit = False; apex = -1e9
    for i in range(220):
        a = seq[i] if i < len(seq) else seq[-1]
        _agent_update(ag, a)
        ball.applyAcceleration(0, GRAV); ball.limitSpeed(0, MAXB); ball.move()
        if ball.isColliding(ag):
            ball.bounce(ag); hit = True
        if ball.isColliding(fs):
            ball.bounce(fs)
        if hit:
            apex = max(apex, ball.y)
        res = ball.checkEdges()
        if res != 0:
            return {"landing_x": ball.x, "scored": res == -1, "hit": hit, "apex": apex}
    return {"landing_x": ball.x, "scored": False, "hit": hit, "apex": apex}


def _score(r, P):
    if not r["hit"]:
        return -200.0
    if r["scored"]:
        depth = P["depth_w"] * min(abs(r["landing_x"]), 18.0)
        apex_pen = P["apex_w"] * max(0.0, r["apex"] - P["apex_ok"])
        return 100.0 + depth - apex_pen
    return -40.0 - r["landing_x"]


def make_policy(seat=0, **kw):
    P = dict(PARAMS); P.update(kw)

    def act(obs):
        sx, sy, svx, svy = obs[0]*10, obs[1]*10, obs[2]*10, obs[3]*10
        bx, by, bvx, bvy = obs[4]*10, obs[5]*10, obs[6]*10, obs[7]*10
        cx, _ = _predict_contact(bx, by, bvx, bvy, P["contact_h"])
        coming = cx is not None and cx > NET_X
        target = (cx + P["hit_offset"]) if coming else P["home_x"]
        target = max(NET_X + 1.6, min(target, WALL - 0.5))
        # contact imminent -> pick the action sequence whose simulated outcome scores best
        if coming and by < P["mpc_h"] and abs(bx - sx) < P["mpc_dx"] and bvy < 6:
            best, best_s = None, -1e9
            for seq in _LIB.values():
                s = _score(_rollout(bx, by, bvx, bvy, sx, sy, svx, svy, seq), P)
                if s > best_s:
                    best_s, best = s, seq
            if best_s > P["attack_floor"]:
                return list(best[0])
        # otherwise position under the predicted landing point (defend first)
        fwd = bwd = 0
        if sx > target + P["deadzone"]:
            fwd = 1
        elif sx < target - P["deadzone"]:
            bwd = 1
        return [fwd, bwd, 0]
    return act


# --- inlined slime-volley physics (Particle + Agent, raw env units) ---
import math as _math


class Particle:
    def __init__(self, x, y, vx, vy, r, c=None):
        self.x = x; self.y = y; self.prev_x = x; self.prev_y = y; self.vx = vx; self.vy = vy; self.r = r
    def move(self):
        self.prev_x = self.x; self.prev_y = self.y
        self.x += self.vx * (1 / 30.0); self.y += self.vy * (1 / 30.0)
    def applyAcceleration(self, ax, ay):
        self.vx += ax * (1 / 30.0); self.vy += ay * (1 / 30.0)
    def limitSpeed(self, mn, mx):
        m2 = self.vx * self.vx + self.vy * self.vy
        if m2 > mx * mx:
            m = _math.sqrt(m2); self.vx = self.vx / m * mx; self.vy = self.vy / m * mx
        if m2 < mn * mn and m2 > 0:
            m = _math.sqrt(m2); self.vx = self.vx / m * mn; self.vy = self.vy / m * mn
    def checkEdges(self):
        r = self.r; REF_W = 48.0; REF_U = 1.5; REF_H = 48.0; F = 1.0; N = 0.1 * (1 / 30.0)
        WALL_W = 1.0; WALL_H = 3.5
        if self.x <= (r - REF_W / 2): self.vx *= -F; self.x = r - REF_W / 2 + N
        if self.x >= (REF_W / 2 - r): self.vx *= -F; self.x = REF_W / 2 - r - N
        if self.y <= WALL_H:
            NETX = WALL_W / 2
            cx = max(-NETX, min(self.x, NETX)); cy = max(REF_U, min(self.y, WALL_H))
            dx = self.x - cx; dy = self.y - cy; d2 = dx * dx + dy * dy
            if d2 < r * r:
                if d2 < 1e-12:
                    side = 1.0 if self.prev_x >= 0 else -1.0
                    self.x = side * (NETX + r) + side * N; self.vx = side * abs(self.vx) * F
                    if self.vy < 0: self.vy = -self.vy * F
                else:
                    d = _math.sqrt(d2); nx = dx / d; ny = dy / d
                    self.x = cx + nx * r; self.y = cy + ny * r
                    vn = self.vx * nx + self.vy * ny
                    self.vx = (self.vx - 2 * vn * nx) * F; self.vy = (self.vy - 2 * vn * ny) * F
        if self.y <= (r + REF_U):
            self.vy *= -F; self.y = r + REF_U + N
            return -1 if self.x <= 0 else 1
        if self.y >= (REF_H - r): self.vy *= -F; self.y = REF_H - r - N
        return 0
    def getDist2(self, p):
        dx = p.x - self.x; dy = p.y - self.y; return dx * dx + dy * dy
    def isColliding(self, p):
        r = self.r + p.r; return r * r > self.getDist2(p)
    def bounce(self, p):
        abx = self.x - p.x; aby = self.y - p.y
        abd = _math.sqrt(abx * abx + aby * aby) or 1e-9
        abx /= abd; aby /= abd; nx = abx; ny = aby
        abx *= 0.1; aby *= 0.1; guard = 0
        while self.isColliding(p) and guard < 100:
            self.x += abx; self.y += aby; guard += 1
        ux = self.vx - p.vx; uy = self.vy - p.vy
        un = ux * nx + uy * ny; ux -= nx * un * 2; uy -= ny * un * 2
        self.vx = ux + p.vx; self.vy = uy + p.vy


class Agent:
    def __init__(self, dir, x, y=1.5, c=None):
        self.dir = dir; self.x = x; self.y = y; self.r = 1.5; self.vx = 0; self.vy = 0
