import numpy as np

GRAVITY = 0.098
DT = 1.0 / 30.0
WALL_X = 2.35
NET_X_MIN = 0.20

PARAMS = dict(
    stand_off=0.050,
    mod_gain=0.050,
    hit_h=0.55,
    jlo=0.42,
    jhi=0.95,
    jclose=0.65,
    jump_vy_max=0.60,
    home=0.35,
    tol=0.030,
    rise_ceiling=0.85,
    deep_x=2.34,
    max_pred=95,
    jump_t_max=22,
)


def _reflect_x(x, vx):
    if x > WALL_X:
        return 2 * WALL_X - x, -vx
    if x < -WALL_X:
        return -2 * WALL_X - x, -vx
    return x, vx


class FlatWallPolicy:
    def __init__(self, seat=0, **params):
        self.seat = seat
        self.p = dict(PARAMS)
        self.p.update(params)

    def reset(self):
        pass

    def predict_hit(self, obs):
        bx, by, bvx, bvy = float(obs[4]), float(obs[5]), float(obs[6]), float(obs[7])
        x, y, vx, vy = bx, by, bvx, bvy
        best = (x, y, vx, vy, 0)
        for t in range(1, int(self.p["max_pred"]) + 1):
            x += vx * DT
            y += vy * DT
            vy -= GRAVITY
            x, vx = _reflect_x(x, vx)
            best = (x, y, vx, vy, t)
            if vy < 0 and y <= self.p["hit_h"] and x > -0.12:
                return best
            if y <= 0.16 and t > 8:
                return best
        return best

    def act(self, obs):
        obs = np.asarray(obs, dtype=float)
        mx, my, mvx, mvy = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        bx, by, bvx, bvy = float(obs[4]), float(obs[5]), float(obs[6]), float(obs[7])
        pred_x, pred_y, pred_vx, pred_vy, pred_t = self.predict_hit(obs)

        # Contact-height-aware offset: stand slightly shallower (closer to the predicted hit)
        # when the ball will be met LOW, where a flatter, safer return reduces conceded points.
        off = self.p["stand_off"] - self.p["mod_gain"] * (self.p["hit_h"] - pred_y)
        off = min(max(off, 0.02), 0.09)

        if pred_x > -0.05 or bx > 0.05 or bvx > 0.15:
            target = pred_x + off
            target = min(max(target, NET_X_MIN + 0.02), self.p["deep_x"])
        else:
            target = self.p["home"]

        if abs(bvx) < 1e-6 and by > 1.0:
            target = max(target, self.p["home"])

        forward = mx > target + self.p["tol"]
        backward = mx < target - self.p["tol"]

        rising_apex = by + max(bvy, 0.0) ** 2 / (2.0 * GRAVITY * 30.0)
        high_rising_skip = bvy > 0.05 and rising_apex > self.p["rise_ceiling"]
        near_pred = abs(pred_x - mx) < self.p["jclose"] and pred_x > -0.20 and pred_t <= self.p["jump_t_max"]
        grounded = my <= 0.18
        jump = (
            grounded
            and not high_rising_skip
            and self.p["jlo"] <= by <= self.p["jhi"]
            and bvy <= self.p["jump_vy_max"]
            and near_pred
        )
        return [int(forward), int(backward), int(jump)]


def make_policy(seat=0):
    policy = FlatWallPolicy(seat=seat)
    return policy.act
