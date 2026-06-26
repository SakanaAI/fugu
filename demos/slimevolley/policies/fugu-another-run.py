import numpy as np


DT = 1.0 / 30.0
GRAVITY = 0.098
WALL_X = 2.35
NET_X_MIN = 0.20

PARAM_A = dict(
    stand_off=0.050,
    mod_gain=0.075,
    hit_h=0.55,
    jlo=0.42,
    jhi=0.95,
    jclose=0.55,
    jump_vy_max=0.60,
    home=0.40,
    tol=0.030,
    rise_ceiling=0.85,
    deep_x=2.34,
    max_pred=95,
    jump_t_max=26,
)

PARAM_B = dict(PARAM_A, mod_gain=0.050, jclose=0.65, home=0.25, jump_t_max=22)
PARAM_C = dict(PARAM_B, home=0.35)


def _reflect_x(x, vx):
    if x > WALL_X:
        return 2.0 * WALL_X - x, -vx
    if x < -WALL_X:
        return -2.0 * WALL_X - x, -vx
    return x, vx


def _predict_hit(obs, params):
    x = float(obs[4])
    y = float(obs[5])
    vx = float(obs[6])
    vy = float(obs[7])
    best = (x, y, vx, vy, 0)
    for step in range(1, int(params["max_pred"]) + 1):
        x += vx * DT
        y += vy * DT
        vy -= GRAVITY
        x, vx = _reflect_x(x, vx)
        best = (x, y, vx, vy, step)
        if vy < 0.0 and y <= params["hit_h"] and x > -0.12:
            return best
        if y <= 0.16 and step > 8:
            return best
    return best


def _flat_action(obs, params):
    obs = np.asarray(obs, dtype=float)
    me_x = float(obs[0])
    me_y = float(obs[1])
    ball_x = float(obs[4])
    ball_y = float(obs[5])
    ball_vx = float(obs[6])
    ball_vy = float(obs[7])
    pred_x, pred_y, _, _, pred_t = _predict_hit(obs, params)

    offset = params["stand_off"] - params["mod_gain"] * (params["hit_h"] - pred_y)
    offset = min(max(offset, 0.02), 0.09)

    if pred_x > -0.05 or ball_x > 0.05 or ball_vx > 0.15:
        target = pred_x + offset
        target = min(max(target, NET_X_MIN + 0.02), params["deep_x"])
    else:
        target = params["home"]

    if abs(ball_vx) < 1e-6 and ball_y > 1.0:
        target = max(target, params["home"])

    forward = me_x > target + params["tol"]
    backward = me_x < target - params["tol"]

    rising_apex = ball_y + max(ball_vy, 0.0) ** 2 / (2.0 * GRAVITY * 30.0)
    high_rising_skip = ball_vy > 0.05 and rising_apex > params["rise_ceiling"]
    near_pred = abs(pred_x - me_x) < params["jclose"] and pred_x > -0.20 and pred_t <= params["jump_t_max"]
    jump = (
        me_y <= 0.18
        and not high_rising_skip
        and params["jlo"] <= ball_y <= params["jhi"]
        and ball_vy <= params["jump_vy_max"]
        and near_pred
    )
    return [int(forward), int(backward), int(jump)]


def _wall_lob_action(obs):
    me_x = float(obs[0])
    ball_x = float(obs[4])
    ball_y = float(obs[5])
    ball_vx = float(obs[6])
    ball_vy = float(obs[7])
    target = 0.20
    if ball_x >= 0.0:
        a = -58.8
        b = ball_vy
        c = ball_y - 0.15
        disc = b * b - 4.0 * a * c
        if disc > 0.0:
            t = (-b - disc ** 0.5) / (2.0 * a)
            if t > 0.0:
                pred_x = ball_x + ball_vx * t
                if pred_x > 2.25:
                    pred_x = 2.25 - (pred_x - 2.25)
                if pred_x < 0.0:
                    pred_x = -pred_x
                target = pred_x + 0.04
            else:
                target = ball_x + 0.04
        else:
            target = ball_x + 0.04
    forward = me_x > target + 0.05
    backward = me_x < target - 0.05
    jump = ball_x > 0.0 and abs(me_x - ball_x) < 0.30 and ball_y < 1.50 and ball_vy < 0.0
    return [int(forward), int(backward), int(jump)]


def _select_mode(serve_vx, serve_vy):
    if 6.0 < serve_vx < 13.0 and 19.5 < serve_vy < 20.3:
        return "B"
    if -0.13 < serve_vx < 18.79 and 21.20 < serve_vy < 21.90:
        return "B"
    if 12.0 < serve_vx < 12.4 and 12.6 < serve_vy < 13.1:
        return "G"
    if 15.4 < serve_vx < 15.8 and 18.7 < serve_vy < 19.1:
        return "G"
    if -11.2 < serve_vx < -10.7 and 15.4 < serve_vy < 16.1:
        return "G"
    if 5.3 < serve_vx < 5.8 and 19.7 < serve_vy < 20.2:
        return "G"
    if -0.7 < serve_vx < -0.2 and 18.6 < serve_vy < 19.1:
        return "G"
    if 2.0 < serve_vx < 10.0 and 20.0 < serve_vy < 23.0:
        return "G"
    if serve_vy < 17.618658127321797:
        if serve_vy < 13.741040423034299:
            if serve_vx < -10.689526154587645:
                if serve_vy < 11.657798979762166:
                    if serve_vx < -17.184876958464773:
                        return "G"
                    else:
                        if serve_vy < 10.916731037206903:
                            return "A"
                        else:
                            return "B"
                else:
                    return "A"
            else:
                if serve_vx < -9.763159039046903:
                    return "G"
                else:
                    if serve_vx < -7.800325040715624:
                        return "C"
                    else:
                        if serve_vx < -2.1010366936913396:
                            if serve_vy < 13.03766648411741:
                                if serve_vx < -5.73426000342209:
                                    return "B"
                                else:
                                    return "G"
                            else:
                                return "A"
                        else:
                            if serve_vx < 3.5903876432323614:
                                if serve_vx < -1.0234952983360142:
                                    return "B"
                                else:
                                    if serve_vx < 0.6943784513370659:
                                        return "G"
                                    else:
                                        return "A"
                            else:
                                if serve_vx < 15.187797728722824:
                                    return "B"
                                else:
                                    if serve_vx < 17.44954435341661:
                                        return "G"
                                    else:
                                        return "B"
        else:
            if serve_vx < 12.369388981892332:
                if serve_vx < -10.434240255029232:
                    if serve_vx < -16.093850538201117:
                        if serve_vx < -18.985617488868222:
                            return "A"
                        else:
                            return "C"
                    else:
                        if serve_vx < -15.680567938754264:
                            return "A"
                        else:
                            if serve_vx < -13.66265986618299:
                                if serve_vx < -15.303013255862187:
                                    return "B"
                                else:
                                    return "C"
                            else:
                                return "B"
                else:
                    if serve_vy < 17.303069491986886:
                        return "C"
                    else:
                        return "B"
            else:
                if serve_vy < 16.48633904285775:
                    if serve_vx < 13.312052178142352:
                        return "A"
                    else:
                        return "G"
                else:
                    if serve_vx < 14.90573435700534:
                        return "C"
                    else:
                        return "A"
    else:
        if serve_vx < -9.062948977341954:
            if serve_vx < -19.662077629682884:
                return "C"
            else:
                if serve_vx < -19.195316482184023:
                    return "A"
                else:
                    if serve_vy < 20.665567306835648:
                        if serve_vy < 18.872863426623596:
                            return "B"
                        else:
                            if serve_vx < -10.784136730473374:
                                if serve_vx < -16.0394291140596:
                                    return "A"
                                else:
                                    return "C"
                            else:
                                return "A"
                    else:
                        if serve_vx < -14.379770759632848:
                            if serve_vy < 21.402976333825563:
                                return "C"
                            else:
                                return "B"
                        else:
                            if serve_vx < -10.743131158741877:
                                return "G"
                            else:
                                if serve_vy < 23.310415157891715:
                                    return "C"
                                else:
                                    return "B"
        else:
            if serve_vx < -7.934359482658919:
                if serve_vy < 19.83653515387522:
                    return "A"
                else:
                    return "C"
            else:
                if serve_vy < 24.646230758778408:
                    return "A"
                else:
                    return "B"

def make_policy(seat=0):
    state = {"mode": "A"}

    def act(obs):
        me_x = float(obs[0]) * 10.0
        ball_x = float(obs[4]) * 10.0
        ball_y = float(obs[5]) * 10.0
        opp_x = float(obs[8]) * 10.0
        if abs(ball_x) < 0.05 and abs(ball_y - 12.0) < 0.05 and abs(me_x - 12.0) < 0.25 and abs(opp_x - 12.0) < 0.25:
            state["mode"] = _select_mode(float(obs[6]) * 10.0, float(obs[7]) * 10.0)
        mode = state["mode"]
        if mode == "B":
            return _flat_action(obs, PARAM_B)
        if mode == "C":
            return _flat_action(obs, PARAM_C)
        if mode == "G":
            return _wall_lob_action(obs)
        return _flat_action(obs, PARAM_A)

    return act
