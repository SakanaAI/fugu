"""Custom Unitree G1 walking controller for MuJoCo (no external control libs).

The robot walks forward at a target speed with a compliant, self-balancing gait.

How it works (all written from scratch -- no locomotion library):
  * Model: MuJoCo Menagerie `unitree_g1/scene.xml`, used VERBATIM. The position
    actuators (kp=500, dampratio=1) and every joint's damping/armature/ctrlrange
    are the model's own defaults. We never edit the XML or scale gains.
  * Balance principle: a biped's floating base is unactuated, so we cannot push
    it directly. Instead the controller shifts the body weight over the stance
    foot *kinematically* -- during a double-support phase both feet stay planted
    and the legs "scissor" to move the pelvis (and hence the CoM) over the next
    stance foot. Only once the CoM is over the stance foot does the swing foot
    lift. This keeps the ZMP inside the support foot, so the robot balances
    itself through ground-reaction forces (no external/balancing forces applied).
  * Footstep plan: a gait clock alternates stance/swing legs; the swing foot
    follows a smooth cycloid to its next footstep; the pelvis target sways
    laterally over the stance foot and advances forward at the target speed.
  * Whole-body IK: a 6-DOF damped-least-squares leg IK (foot position + flat
    sole) converts pelvis+foot targets into joint position targets.
  * Actuation: the ONLY thing written to the sim is `data.ctrl` for the existing
    actuators. `qfrc_applied` / `xfrc_applied` are never touched (verified == 0).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict
import numpy as np
import mujoco

DEFAULT_MODEL = "third_party/mujoco_menagerie/unitree_g1/scene.xml"


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def quat_to_rpy(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class LegIK:
    """6-DOF damped-least-squares IK for both legs (position + flat sole)."""

    def __init__(self, model):
        self.m = model
        self.ik = mujoco.MjData(model)
        JN = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
        self.qadr = {n: model.jnt_qposadr[JN[n]] for n in JN}
        self.dadr = {n: model.jnt_dofadr[JN[n]] for n in JN}
        self.jr = {n: model.jnt_range[JN[n]].copy() for n in JN}
        self.LEGS = {
            'left': ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
                     'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint'],
            'right': ['right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
                      'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'],
        }
        self.SITE = {'left': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'left_foot'),
                     'right': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_foot')}
        self.Rt = np.eye(3)  # desired foot orientation: flat, facing +x

    def solve(self, base_pos, base_quat, foot_tgt, seed, iters=40):
        ik, m = self.ik, self.m
        ik.qpos[:] = seed
        ik.qpos[:3] = base_pos
        ik.qpos[3:7] = base_quat
        # Break the straight-leg singularity so vertical foot motion is well-posed.
        for leg in ('left', 'right'):
            kq = self.qadr[f'{leg}_knee_joint']
            if ik.qpos[kq] < 0.2:
                ik.qpos[kq] = 0.3
        for _ in range(iters):
            mujoco.mj_kinematics(m, ik)
            mujoco.mj_comPos(m, ik)
            worst = 0.0
            for leg in ('left', 'right'):
                cols = np.array([self.dadr[j] for j in self.LEGS[leg]])
                sid = self.SITE[leg]
                perr = foot_tgt[leg] - ik.site_xpos[sid]
                Rc = ik.site_xmat[sid].reshape(3, 3)
                Re = self.Rt @ Rc.T
                rerr = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
                worst = max(worst, np.linalg.norm(perr))
                jp = np.zeros((3, m.nv))
                jr = np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, ik, jp, jr, sid)
                J = np.vstack([jp[:, cols], jr[:, cols]])
                e = np.concatenate([perr, rerr])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e) * 0.8
                dq = np.clip(dq, -0.25, 0.25)
                for j, dd in zip(self.LEGS[leg], dq):
                    qi = self.qadr[j]
                    lo, hi = self.jr[j]
                    ik.qpos[qi] = np.clip(ik.qpos[qi] + dd, lo + 1e-3, hi - 1e-3)
            if worst < 5e-4:
                break
        return ik.qpos.copy()


class Walker:
    def __init__(self, model, data, p):
        self.m = model
        self.p = p
        self.dt = model.opt.timestep
        self.ik = LegIK(model)
        self.key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        self.act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
        self.cmin = model.actuator_ctrlrange[:, 0].copy()
        self.cmax = model.actuator_ctrlrange[:, 1].copy()
        self.pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.qadr = self.ik.qadr
        self.H = p['height']
        self.half = p['half_width']
        self.footz = p['foot_z']

        # Build the initial compliant crouch via IK and start the robot in it.
        seed = model.key_qpos[self.key].copy()
        ft0 = {'left': np.array([0.0, self.half, self.footz]),
               'right': np.array([0.0, -self.half, self.footz])}
        q0 = self.ik.solve(np.array([0, 0, self.H]), np.array([1, 0, 0, 0]), ft0, seed, iters=300)
        data.qpos[:] = q0
        data.qpos[:3] = [0, 0, self.H]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        self.q_seed = q0.copy()

        # Nominal ctrl vector (legs from IK, arms/waist set to a natural posture).
        self.ctrl = model.key_ctrl[self.key].copy()
        for leg in ('left', 'right'):
            for j in self.ik.LEGS[leg]:
                if j in self.act:
                    self.ctrl[self.act[j]] = q0[self.qadr[j]]
        for n, v in [('left_shoulder_pitch_joint', 0.25), ('left_shoulder_roll_joint', 0.16), ('left_elbow_joint', 0.9),
                     ('right_shoulder_pitch_joint', 0.25), ('right_shoulder_roll_joint', -0.16), ('right_elbow_joint', 0.9),
                     ('waist_yaw_joint', 0.0), ('waist_roll_joint', 0.0), ('waist_pitch_joint', 0.0)]:
            if n in self.act:
                self.ctrl[self.act[n]] = v

        # Gait timing / state.
        self.t0 = p['start_time']
        self.t_shift = p['shift_time']
        self.Ts = p['step_time']
        self.dsf = p['ds_frac']
        self.sway = p['sway']
        self.L = p['vx'] * p['step_time']
        self.foot_world = {'left': np.array([0.0, self.half, self.footz]),
                           'right': np.array([0.0, -self.half, self.footz])}
        self.cur_step = None
        self.swing = 'right'
        self.swing_from = self.foot_world['right'].copy()
        self.swing_to = self.foot_world['right'].copy()
        self.px = 0.0
        self.py = 0.0
        self.sway_prev = 0.0

    def compute(self, data, t):
        p = self.p
        des = self.ctrl.copy()
        base_quat = np.array([1.0, 0, 0, 0])

        tw = t - self.t0
        if tw < 0.0:
            ft = {'left': self.foot_world['left'], 'right': self.foot_world['right']}
            base_pos = np.array([0.0, 0.0, self.H])
            ramp = 0.0
        else:
            ramp = smoothstep(tw / p['ramp_time'])
            tshift = t - self.t0          # time since start
            if tshift < self.t_shift:
                # Phase A: shift weight onto the left foot (first stance), no step.
                s = smoothstep(tshift / self.t_shift)
                self.py = s * (self.sway)          # +y over left foot
                self.px = 0.0
                ft = {'left': self.foot_world['left'], 'right': self.foot_world['right']}
                base_pos = np.array([self.px, self.py, self.H])
                self.sway_prev = self.py
            else:
                # Phase B: walking.
                twb = tshift - self.t_shift
                step = int(twb / self.Ts)
                frac = twb / self.Ts - step
                swing = 'right' if step % 2 == 0 else 'left'
                stance = 'left' if swing == 'right' else 'right'
                if step != self.cur_step:
                    # The previously swinging foot has landed: commit it.
                    if self.cur_step is not None:
                        self.foot_world[self.swing] = self.swing_to.copy()
                    self.cur_step = step
                    self.swing = swing
                    self.swing_from = self.foot_world[swing].copy()
                    sign = self.half if swing == 'left' else -self.half
                    self.swing_to = np.array([self.px + self.L + p['step_ahead'], sign, self.footz])
                    self.sway_prev = self.py

                # Advance planned pelvis forward at target speed.
                self.px += p['vx'] * self.dt

                # Lateral sway: move CoM over the current stance foot.
                stance_sway = (self.sway if stance == 'left' else -self.sway)
                if frac < self.dsf:
                    s = smoothstep(frac / self.dsf)
                    self.py = (1 - s) * self.sway_prev + s * stance_sway
                else:
                    self.py = stance_sway

                # Swing foot trajectory (only after the double-support shift).
                if frac < self.dsf:
                    ft_sw = self.swing_from.copy()
                else:
                    st = (frac - self.dsf) / (1.0 - self.dsf)
                    ss = smoothstep(st)
                    x = (1 - ss) * self.swing_from[0] + ss * self.swing_to[0]
                    y = (1 - ss) * self.swing_from[1] + ss * self.swing_to[1]
                    z = self.footz + p['clearance'] * math.sin(math.pi * st)
                    ft_sw = np.array([x, y, z])
                ft = {stance: self.foot_world[stance].copy(), swing: ft_sw}
                base_pos = np.array([self.px, self.py, self.H])

        # Whole-body IK -> joint position targets.
        qsol = self.ik.solve(base_pos, base_quat, ft, self.q_seed, iters=p['ik_iters'])
        self.q_seed = qsol.copy()
        for leg in ('left', 'right'):
            for j in self.ik.LEGS[leg]:
                if j in self.act:
                    tgt = qsol[self.qadr[j]]
                    nom = self.ctrl[self.act[j]]
                    des[self.act[j]] = (1 - ramp) * nom + ramp * tgt

        # Arm counter-swing for natural look + angular-momentum cancelation.
        if tw >= self.t0 and t - self.t0 > self.t_shift:
            phi = (t - self.t0 - self.t_shift) / (2 * self.Ts)
            arm = ramp * p['arm_amp'] * math.sin(2 * math.pi * phi)
            if 'left_shoulder_pitch_joint' in self.act:
                des[self.act['left_shoulder_pitch_joint']] = 0.25 + arm
            if 'right_shoulder_pitch_joint' in self.act:
                des[self.act['right_shoulder_pitch_joint']] = 0.25 - arm

        return np.clip(des, self.cmin, self.cmax)


def simulate(model_path, p, duration, video_path, metrics_path, width, height, fps):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    w = Walker(model, data, p)

    renderer = writer = None
    if video_path:
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.azimuth = 130
    cam.elevation = -12
    cam.lookat[:] = [0, 0, 0.55]

    start_x = float(data.qpos[0])
    minz = 9.0
    maxt = 0.0
    fallen = False
    samples = []
    next_frame = 0.0
    fp = 1.0 / fps
    walk_dist = 0.0
    steps = int(duration / model.opt.timestep)
    for s in range(steps):
        t = float(data.time)
        data.ctrl[:] = w.compute(data, t)
        mujoco.mj_step(model, data)
        r, pi, ya = quat_to_rpy(data.qpos[3:7])
        tilt = max(abs(r), abs(pi))
        minz = min(minz, float(data.qpos[2]))
        maxt = max(maxt, tilt)
        if data.qpos[2] < 0.45 or tilt > 1.0:
            fallen = True
        if s % int(0.5 / model.opt.timestep) == 0:
            samples.append({"t": round(t, 3), "x": round(float(data.qpos[0]), 3),
                            "y": round(float(data.qpos[1]), 3), "z": round(float(data.qpos[2]), 3),
                            "vx": round(float(data.qvel[0]), 3), "roll": round(r, 3),
                            "pitch": round(pi, 3), "ncon": int(data.ncon)})
        if renderer is not None and t + 1e-12 >= next_frame:
            cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.5]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            next_frame += fp
    if writer:
        writer.close()
    if renderer:
        renderer.close()

    # Speed measured over the steady walking window (exclude start-up & shift).
    t_walk_start = p['start_time'] + p['shift_time']
    walk_samples = [s for s in samples if s['t'] >= t_walk_start]
    if len(walk_samples) >= 2:
        x0 = walk_samples[0]['x']; x1 = walk_samples[-1]['x']
        dtw = walk_samples[-1]['t'] - walk_samples[0]['t']
        steady_speed = (x1 - x0) / dtw if dtw > 0 else 0.0
    else:
        steady_speed = 0.0
    dist = float(data.qpos[0] - start_x)
    res = {
        "model_path": model_path, "params": p,
        "duration_s": round(float(data.time), 3),
        "distance_m": round(dist, 3),
        "average_speed_mps": round(dist / float(data.time), 4),
        "steady_speed_mps": round(steady_speed, 4),
        "target_speed_mps": p['vx'],
        "final_xyz": [round(float(x), 3) for x in data.qpos[:3]],
        "min_base_z_m": round(minz, 3),
        "max_tilt_rad": round(maxt, 3),
        "fallen": fallen,
        "video": video_path,
        "video_size": [width, height] if video_path else None,
        "fps": fps if video_path else None,
        "xfrc_applied_norm": float(np.linalg.norm(data.xfrc_applied)),
        "qfrc_applied_norm": float(np.linalg.norm(data.qfrc_applied)),
        "samples": samples,
    }
    txt = json.dumps(res, indent=2)
    print(txt)
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(txt + "\n")
    return res


def build_params(a):
    return dict(vx=a.vx, height=a.height_com, half_width=a.half_width, foot_z=a.foot_z,
                step_time=a.step_time, ds_frac=a.ds_frac, sway=a.sway, clearance=a.clearance,
                step_ahead=a.step_ahead, arm_amp=a.arm_amp, ramp_time=a.ramp_time,
                start_time=a.start_time, shift_time=a.shift_time, ik_iters=a.ik_iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--height-com", type=float, default=0.68)
    ap.add_argument("--half-width", type=float, default=0.12)
    ap.add_argument("--foot-z", type=float, default=0.0331)
    ap.add_argument("--step-time", type=float, default=0.40)
    ap.add_argument("--ds-frac", type=float, default=0.30)
    ap.add_argument("--sway", type=float, default=0.10)
    ap.add_argument("--clearance", type=float, default=0.05)
    ap.add_argument("--step-ahead", type=float, default=0.0)
    ap.add_argument("--arm-amp", type=float, default=0.25)
    ap.add_argument("--ramp-time", type=float, default=1.0)
    ap.add_argument("--start-time", type=float, default=0.5)
    ap.add_argument("--shift-time", type=float, default=0.6)
    ap.add_argument("--ik-iters", type=int, default=30)
    a = ap.parse_args()
    simulate(a.model, build_params(a), a.duration, a.video, a.metrics, a.width, a.height, a.fps)


if __name__ == "__main__":
    main()
