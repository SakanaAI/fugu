#!/usr/bin/env python3
"""Custom DCM/capture-point marching controller for Unitree G1 in MuJoCo.

This is a project script derived from the verified scratch harness. It uses the
downloaded Unitree G1/MuJoCo model unchanged, writes only existing actuator
controls (`data.ctrl`), and applies no external forces.

Controller summary
------------------
* Start in the verified bent-knee crouch (built by Walker.__init__).
* INITIATION (priority #1): min-jerk lateral shift of the IK pelvis-y target toward
  the first stance foot, ending with ~zero velocity (min-jerk has zero end-vel),
  then a short hold so the CoM settles over the stance foot before the first lift.
* STEP: fixed-period OR event-driven (touchdown) stepping. Each step optionally has
  a double-support fraction (both feet planted) to re-center, then a single-support
  swing along a sine-z cycloid to the DCM/capture-point foot target.
* DCM foot placement (priority, capture-point law):
      mj_subtreeVel -> com = subtree_com[pid], comv = subtree_linvel[pid]
      h_com = com_z - stance_foot_site_z          (MEASURED, not pelvis height)
      omega = sqrt(9.81/h_com)
      DCM_y = com_y + comv_y/omega
      b     = d*(1 - tanh(omega*T/2))
      y_swing_target = DCM_y + sign_swing*b        (blended w/ nominal +/-d via gain)
* CoM height held constant by commanding IK base z = pelvis_height every tick.
* Only data.ctrl is written. qfrc_applied / xfrc_applied are never touched.
"""
from __future__ import annotations
import argparse, itertools, json, math, random, sys, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import mujoco

ROOT = "/Users/yujintang/Projects/fugu/demos/humanoid_control/fugu_ultra"
sys.path.insert(0, ROOT)
from scripts.walk_g1 import Walker, LegIK, quat_to_rpy, DEFAULT_MODEL  # noqa: E402


def minjerk(u: float) -> float:
    """5th-order min-jerk 0->1 with zero end velocity & accel (settles cleanly)."""
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@dataclass
class Cfg:
    # ---- primary sweep knobs (the success-metric tuple) ----
    T: float = 0.40                 # step period (s)
    d: float = 0.11                 # stance half-width (foot y = +/- d)
    pelvis_height: float = 0.72     # commanded IK base z (constant CoM height)
    clearance: float = 0.03         # swing-foot peak lift (m)
    # ---- initiation scheme ----
    init_delay: float = 0.10        # quiet settle in crouch before shifting
    init_shift: float = 0.90        # duration of min-jerk lateral CoM shift (s)
    init_hold: float = 0.25         # hold over stance foot to kill residual vel (s)
    init_frac: float = 0.85         # pelvis-y at end of init = init_frac * d
    first_stance: str = "left"
    # ---- single-support lateral target ----
    sway_frac: float = 0.85         # SS pelvis-y target = +/- sway_frac * d
    ds_frac: float = 0.30           # start-of-step double-support fraction
    # ---- DCM capture-point foot placement ----
    dcm_gain: float = 1.0           # blend nominal->capture target
    dcm_clip: float = 0.05          # max |deviation| of swing-y from nominal +/-d
    min_abs_y: float = 0.07         # never cross feet closer than this
    # ---- stepping mode ----
    event_driven: int = 0           # 1 = step on touchdown (+ max-time guard)
    td_min_frac: float = 0.55       # earliest touchdown check (frac of T) in event mode
    # ---- sagittal capture (small; vx=0) ----
    x_gain: float = 0.0
    x_clip: float = 0.04
    # ---- numerics ----
    foot_z: float = 0.0331
    control_dt: float = 0.005
    ik_iters: int = 20
    arm_amp: float = 0.05


class Marcher:
    def __init__(self, model, data, cfg: Cfg):
        self.m = model
        self.cfg = cfg
        p = dict(vx=0.0, height=cfg.pelvis_height, half_width=cfg.d, foot_z=cfg.foot_z,
                 step_time=cfg.T, ds_frac=cfg.ds_frac, sway=cfg.sway_frac * cfg.d,
                 clearance=cfg.clearance, step_ahead=0.0, arm_amp=cfg.arm_amp,
                 ramp_time=0.1, start_time=0.0, shift_time=cfg.init_shift, ik_iters=cfg.ik_iters)
        self.w0 = Walker(model, data, p)           # builds verified crouch into `data`
        self.ik: LegIK = self.w0.ik
        self.act = self.w0.act
        self.qadr = self.w0.qadr
        self.ctrl0 = self.w0.ctrl.copy()
        self.cmin, self.cmax = self.w0.cmin, self.w0.cmax
        self.pid = self.w0.pid
        self.site = self.ik.SITE
        self.q_seed = self.w0.q_seed.copy()
        self.bL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
        self.bR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_ankle_roll_link')

        self.nominal = {'left': np.array([0.0,  cfg.d, cfg.foot_z]),
                        'right': np.array([0.0, -cfg.d, cfg.foot_z])}
        self.foot_world = {k: v.copy() for k, v in self.nominal.items()}
        self.first_sign = 1.0 if cfg.first_stance == 'left' else -1.0
        self.gait_start = cfg.init_delay + cfg.init_shift + cfg.init_hold
        # stepping state
        self.step_idx = -1
        self.swing = None
        self.stance = cfg.first_stance
        self.swing_from = None
        self.swing_to = None
        self.prev_sway = self.first_sign * cfg.init_frac * cfg.d
        self.step_t0 = self.gait_start
        self.last_update_t = -1e9
        self.last_ctrl = self.ctrl0.copy()
        data.ctrl[:] = self.ctrl0
        mujoco.mj_forward(model, data)

    # ---- CoM state (lag-free) ----
    def com_state(self, data):
        mujoco.mj_subtreeVel(self.m, data)
        return data.subtree_com[self.pid].copy(), data.subtree_linvel[self.pid].copy()

    def stance_in_contact(self, data, stance):
        bid = self.bL if stance == 'left' else self.bR
        for c in range(data.ncon):
            for g in (data.contact[c].geom1, data.contact[c].geom2):
                if self.m.geom_bodyid[g] == bid:
                    return True
        return False

    def begin_step(self, data, idx):
        cfg = self.cfg
        swing = ('right' if idx % 2 == 0 else 'left') if cfg.first_stance == 'left' \
                else ('left' if idx % 2 == 0 else 'right')
        stance = 'left' if swing == 'right' else 'right'
        if self.swing is not None:
            self.foot_world[self.swing] = self.swing_to.copy()  # commit landed foot
        self.step_idx = idx
        self.swing, self.stance = swing, stance
        self.swing_from = self.foot_world[swing].copy()
        self.swing_to = self.nominal[swing].copy()

    def targets(self, data, t):
        """Return (base_pos, foot_tgt, phase, dbg)."""
        cfg = self.cfg
        H = cfg.pelvis_height
        ft = {k: v.copy() for k, v in self.foot_world.items()}
        dbg = dict(omega=float('nan'), b=float('nan'), dcm_y=float('nan'), stance=self.stance)

        # ---------- INITIATION ----------
        if t < cfg.init_delay:
            return np.array([0.0, 0.0, H]), ft, 'delay', dbg
        if t < cfg.init_delay + cfg.init_shift:
            u = (t - cfg.init_delay) / max(1e-9, cfg.init_shift)
            by = self.first_sign * cfg.init_frac * cfg.d * minjerk(u)
            return np.array([0.0, by, H]), ft, 'init', dbg
        if t < self.gait_start:
            by = self.first_sign * cfg.init_frac * cfg.d
            return np.array([0.0, by, H]), ft, 'hold', dbg

        # ---------- WALKING ----------
        if self.step_idx < 0:
            self.begin_step(data, 0)
            self.step_t0 = t

        if cfg.event_driven:
            frac = (t - self.step_t0) / cfg.T
            if frac >= 1.0 or (frac >= cfg.td_min_frac and frac >= cfg.ds_frac
                               and self.stance_in_contact(data, self.swing)
                               and (t - self.step_t0) > cfg.ds_frac * cfg.T + 0.04):
                self.prev_sway = self._sway_target(self.stance)
                self.begin_step(data, self.step_idx + 1)
                self.step_t0 = t
            frac = min(0.999, (t - self.step_t0) / cfg.T)
        else:
            idx = int((t - self.gait_start) / cfg.T)
            if idx != self.step_idx:
                self.prev_sway = self._sway_target(self.stance)
                self.begin_step(data, idx)
                self.step_t0 = self.gait_start + idx * cfg.T
            frac = (t - self.step_t0) / cfg.T
            frac = min(0.999, max(0.0, frac))

        stance, swing = self.stance, self.swing
        sign_sw = 1.0 if swing == 'left' else -1.0
        ds = max(0.0, min(0.9, cfg.ds_frac))
        stance_y = self._sway_target(stance)

        # lateral pelvis target: ramp from previous sway to current stance side in DS,
        # then hold over stance foot through single support.
        if frac < ds and ds > 1e-6:
            s = minjerk(frac / ds)
            by = (1 - s) * self.prev_sway + s * stance_y
        else:
            by = stance_y

        phase = 'ds'
        if frac >= ds:
            st = (frac - ds) / max(1e-9, 1.0 - ds)
            com, comv = self.com_state(data)
            footz = float(data.site_xpos[self.site[stance], 2])
            h = max(0.30, com[2] - footz)
            omega = math.sqrt(9.81 / h)
            dcm_y = float(com[1] + comv[1] / omega)
            b = cfg.d * (1.0 - math.tanh(omega * cfg.T / 2.0))
            y_cap = dcm_y + sign_sw * b
            y_nom = sign_sw * cfg.d
            y_tgt = y_nom + cfg.dcm_gain * (y_cap - y_nom)
            y_tgt = max(min(y_tgt, y_nom + cfg.dcm_clip), y_nom - cfg.dcm_clip)
            y_tgt = sign_sw * max(cfg.min_abs_y, min(abs(y_tgt), cfg.d + cfg.dcm_clip))
            x_cap = float(com[0] + comv[0] / omega)
            x_tgt = max(-cfg.x_clip, min(cfg.x_clip, cfg.x_gain * x_cap))
            self.swing_to = np.array([x_tgt, y_tgt, cfg.foot_z])
            ss = minjerk(st)
            x = (1 - ss) * self.swing_from[0] + ss * self.swing_to[0]
            y = (1 - ss) * self.swing_from[1] + ss * self.swing_to[1]
            z = cfg.foot_z + cfg.clearance * math.sin(math.pi * st)
            ft[swing] = np.array([x, y, z])
            phase = 'swing'
            dbg.update(omega=omega, b=b, dcm_y=dcm_y, stance=stance)
        ft[stance] = self.foot_world[stance].copy()
        return np.array([0.0, by, H]), ft, phase, dbg

    def _sway_target(self, stance):
        return (1.0 if stance == 'left' else -1.0) * self.cfg.sway_frac * self.cfg.d

    def compute(self, data, t):
        cfg = self.cfg
        if t - self.last_update_t + 1e-12 < cfg.control_dt:
            return self.last_ctrl
        self.last_update_t = t
        base_pos, ft, phase, dbg = self.targets(data, t)
        qsol = self.ik.solve(base_pos, np.array([1.0, 0, 0, 0]), ft, self.q_seed, iters=cfg.ik_iters)
        self.q_seed = qsol.copy()
        des = self.ctrl0.copy()
        for leg in ('left', 'right'):
            for j in self.ik.LEGS[leg]:
                if j in self.act:
                    des[self.act[j]] = qsol[self.qadr[j]]
        if t >= self.gait_start and cfg.arm_amp:
            phi = (t - self.gait_start) / (2.0 * cfg.T)
            arm = cfg.arm_amp * math.sin(2.0 * math.pi * phi)
            if 'left_shoulder_pitch_joint' in self.act:
                des[self.act['left_shoulder_pitch_joint']] = 0.25 + arm
            if 'right_shoulder_pitch_joint' in self.act:
                des[self.act['right_shoulder_pitch_joint']] = 0.25 - arm
        self.last_ctrl = np.clip(des, self.cmin, self.cmax)
        return self.last_ctrl


def run_one(cfg: Cfg, duration: float = 6.0, stop_on_fall: bool = True, log: bool = False) -> dict:
    model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
    data = mujoco.MjData(model)
    mc = Marcher(model, data, cfg)
    dt = model.opt.timestep
    N = int(duration / dt)
    minz = 9.0; max_tilt = 0.0; fall_time = None
    maxlift = {'left': 0.0, 'right': 0.0}
    com_y_min = 9.0; com_y_max = -9.0; com_z0 = None; com_z_drop = 0.0
    both = sgl = 0
    samples = []
    for s in range(N):
        t = float(data.time)
        data.ctrl[:] = mc.compute(data, t)
        mujoco.mj_step(model, data)
        r, pi, _ = quat_to_rpy(data.qpos[3:7])
        tilt = max(abs(r), abs(pi))
        max_tilt = max(max_tilt, tilt)
        z = float(data.qpos[2]); minz = min(minz, z)
        mujoco.mj_subtreeVel(model, data); com = data.subtree_com[mc.pid]
        if com_z0 is None: com_z0 = float(com[2])
        com_z_drop = max(com_z_drop, com_z0 - float(com[2]))
        com_y_min = min(com_y_min, float(com[1])); com_y_max = max(com_y_max, float(com[1]))
        for leg in ('left', 'right'):
            maxlift[leg] = max(maxlift[leg], float(data.site_xpos[mc.site[leg], 2]) - cfg.foot_z)
        if t >= mc.gait_start:
            cL = mc.stance_in_contact(data, 'left'); cR = mc.stance_in_contact(data, 'right')
            if cL and cR: both += 1
            elif cL != cR: sgl += 1
        if log and s % max(1, int(0.25 / dt)) == 0:
            _, _, phase, dbg = mc.targets(data, t)
            samples.append(dict(t=round(t, 3), z=round(z, 3), comy=round(float(com[1]), 4),
                                tilt=round(float(tilt), 4), phase=phase, stance=dbg['stance'],
                                ncon=int(data.ncon)))
        if (z < 0.45 or tilt > 1.0) and fall_time is None:
            fall_time = float(data.time)
            if stop_on_fall:
                break
    walk_steps = both + sgl
    survived = fall_time is None and float(data.time) >= duration - 1e-9
    res = dict(
        survived=survived, sim_time=round(float(data.time), 3),
        fall_time=None if fall_time is None else round(fall_time, 3),
        final_base_z=round(float(data.qpos[2]), 4), min_base_z=round(minz, 4),
        max_tilt=round(float(max_tilt), 4),
        single_support_frac=round(sgl / walk_steps, 3) if walk_steps else 0.0,
        double_support_frac=round(both / walk_steps, 3) if walk_steps else 0.0,
        max_foot_lift_mm={k: round(v * 1000, 1) for k, v in maxlift.items()},
        com_y_excursion=[round(com_y_min, 4), round(com_y_max, 4)],
        com_z_drop=round(com_z_drop, 4),
        qfrc_norm=float(np.linalg.norm(data.qfrc_applied)),
        xfrc_norm=float(np.linalg.norm(data.xfrc_applied)),
        cfg=asdict(cfg))
    if log:
        res['samples'] = samples
    return res


def cfg_from_args(a) -> Cfg:
    return Cfg(T=a.T, d=a.d, pelvis_height=a.pelvis_height, clearance=a.clearance,
               init_delay=a.init_delay, init_shift=a.init_shift, init_hold=a.init_hold,
               init_frac=a.init_frac, first_stance=a.first_stance, sway_frac=a.sway_frac,
               ds_frac=a.ds_frac, dcm_gain=a.dcm_gain, dcm_clip=a.dcm_clip, min_abs_y=a.min_abs_y,
               event_driven=a.event_driven, td_min_frac=a.td_min_frac, x_gain=a.x_gain,
               x_clip=a.x_clip, foot_z=a.foot_z, control_dt=a.control_dt, ik_iters=a.ik_iters,
               arm_amp=a.arm_amp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=6.0)
    ap.add_argument('--log', action='store_true')
    for name, val in [('--T', 0.40), ('--d', 0.11), ('--pelvis-height', 0.72), ('--clearance', 0.03),
                      ('--init-delay', 0.10), ('--init-shift', 0.90), ('--init-hold', 0.25),
                      ('--init-frac', 0.85), ('--sway-frac', 0.85), ('--ds-frac', 0.30),
                      ('--dcm-gain', 1.0), ('--dcm-clip', 0.05), ('--min-abs-y', 0.07),
                      ('--td-min-frac', 0.55), ('--x-gain', 0.0), ('--x-clip', 0.04),
                      ('--foot-z', 0.0331), ('--control-dt', 0.005), ('--arm-amp', 0.05)]:
        ap.add_argument(name, type=float, default=val)
    ap.add_argument('--first-stance', default='left', choices=['left', 'right'])
    ap.add_argument('--event-driven', type=int, default=0)
    ap.add_argument('--ik-iters', type=int, default=20)
    a = ap.parse_args()
    res = run_one(cfg_from_args(a), duration=a.duration, stop_on_fall=False, log=a.log)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
