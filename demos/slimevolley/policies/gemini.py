def make_policy(seat=0):
    def act(obs):
        agent_x, agent_y = obs[0], obs[1]
        ball_x, ball_y, ball_vx, ball_vy = obs[4], obs[5], obs[6], obs[7]
        GRAVITY = -117.6
        offset = 0.04
        jump_y = 1.5
        target_x = 0.20
        if ball_x < 0: target_x = 0.20
        else:
            a = 0.5 * GRAVITY; b = ball_vy; c = ball_y - 0.15
            disc = b**2 - 4*a*c
            if disc > 0:
                t = (-b - disc**0.5) / (2*a)
                if t > 0:
                    pred_x = ball_x + ball_vx * t
                    if pred_x > 2.25: pred_x = 2.25 - (pred_x - 2.25)
                    if pred_x < 0: pred_x = -pred_x
                    target_x = pred_x + offset
                else: target_x = ball_x + offset
            else: target_x = ball_x + offset
        action = [0, 0, 0]
        if agent_x < target_x - 0.05: action[1] = 1
        elif agent_x > target_x + 0.05: action[0] = 1
        if ball_x > 0 and abs(agent_x - ball_x) < 0.3 and ball_y < jump_y and ball_vy < 0: action[2] = 1
        return action
    return act
