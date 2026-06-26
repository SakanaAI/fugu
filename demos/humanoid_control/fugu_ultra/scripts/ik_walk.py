"""Custom Unitree G1 walking controller for MuJoCo.

This controller intentionally does not use a locomotion/control library.  It uses
an internal gait clock, a small linear-inverted-pendulum style footstep schedule,
and damped-least-squares leg IK using MuJoCo's analytic site Jacobians.  The only
commands sent to MuJoCo are the model's existing position actuator controls.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import imageio.v2 as imageio
import mujoco
import numpy as np

MODEL = "third_party/mujoco_menagerie/unitree_g1/scene.xml"
VIDEO = "artifacts/g1_walk_10s_640x480.mp4"
METRICS = "artifacts/g1_walk_metrics.json"


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def quat_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def yaw_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class FootstepPlanner:
    """Alternating footstep planner in a virtual pelvis frame."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, target_speed: float = 0.5):
        self.v = target_speed
        self.start_time = 0.7
        self.ramp_time = 1.2
        self.step_time = 0.52              # one footfall every 0.52 s => 0.26 m steps for 0.5 m/s
        self.swing_time = 0.34
        self.step_length = self.v * self.step_time
        self.clearance = 0.055
        self.lateral_com = 0.055
        self.foot_width = 0.118506455
        self.site_z = 0.0331362476
        self.max_rel_x = 0.32
        self.nom_height = 0.785

        self.sid = {
            "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
            "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
        }
        mujoco.mj_forward(model, data)
        self.initial_rel = {
            "left": data.site_xpos[self.sid["left"]].copy() - data.qpos[:3].copy(),
            "right": data.site_xpos[self.sid["right"]].copy() - data.qpos[:3].copy(),
        }
        self.initial_rel["left"][2] = self.site_z - data.qpos[2]
        self.initial_rel["right"][2] = self.site_z - data.qpos[2]

    def phase_info(self, t: float) -> Tuple[float, float, str, float]:
        tg = max(0.0, t - self.start_time)
        step_idx = int(tg // self.step_time)
        tau = tg - step_idx * self.step_time
        # Right foot takes the first step so left starts as stance.
        swing_leg = "right" if step_idx % 2 == 0 else "left"
        swing_u = tau / self.swing_time if tau < self.swing_time else 1.0
        ramp = smoothstep(min(1.0, tg / self.ramp_time))
        return tg, tau, swing_leg, ramp if t >= self.start_time else 0.0

    def foot_virtual(self, leg: str, t: float) -> np.ndarray:
        """Virtual foot-site position relative to start-world coordinates."""
        tg, tau, swing_leg, ramp = self.phase_info(t)
        if ramp <= 0.0:
            # Initial feet are under the pelvis.
            y = self.foot_width if leg == "left" else -self.foot_width
            return np.array([0.0, y, self.site_z])

        step_idx = int(tg // self.step_time)
        # Count completed swing events for each leg. Right swings on even indices, left on odd.
        # Each foot placement advances by 2*step_length for that same foot.
        if leg == "right":
            completed = (step_idx + 1) // 2
            current_x = completed * 2.0 * self.step_length
            next_x = current_x + 2.0 * self.step_length
            is_swing = (swing_leg == "right" and tau < self.swing_time)
        else:
            completed = step_idx // 2
            current_x = completed * 2.0 * self.step_length + self.step_length
            # Before the first left step, keep left at x=0 instead of x=step_length.
            if step_idx == 0:
                current_x = 0.0
            next_x = current_x + 2.0 * self.step_length
            is_swing = (swing_leg == "left" and tau < self.swing_time)
        y = self.foot_width if leg == "left" else -self.foot_width
        z = self.site_z
        if is_swing:
            u = smoothstep(tau / self.swing_time)
            x0 = current_x
            x1 = next_x
            z += self.clearance * math.sin(math.pi * (tau / self.swing_time))
            x = (1.0 - u) * x0 + u * x1
        else:
            x = current_x
        return np.array([x, y, z])

    def desired_relative_feet(self, data: mujoco.MjData, t: float) -> Dict[str, np.ndarray]:
        tg, tau, swing_leg, ramp = self.phase_info(t)
        # Virtual pelvis trajectory.  The lateral COM shift is deliberately small
        # and smoothed; it biases the pelvis over the current stance foot without
        # enforcing a rigid kinematic body path.
        x_v = self.v * tg * ramp
        if ramp <= 0.0:
            y_v = 0.0
        else:
            # shift toward the stance side; positive y=left.
            stance_side = +1.0 if swing_leg == "right" else -1.0
            step_phase = min(1.0, tau / self.step_time)
            y_v = ramp * stance_side * self.lateral_com * math.sin(math.pi * step_phase)
        z_v = self.nom_height - 0.015 * ramp * math.sin(2.0 * math.pi * (tg / (2.0 * self.step_time))) ** 2
        pelvis_v = np.array([x_v, y_v, z_v])
        rel = {}
        for leg in ("left", "right"):
            r = self.foot_virtual(leg, t) - pelvis_v
            # Do not ask the leg to overextend; if the real robot lags behind the
            # virtual plan, this cap softens the request instead of forcing a fall.
            r[0] = max(-self.max_rel_x, min(self.max_rel_x, r[0]))
            # Blend in from the exact stand pose at startup.
            r = (1.0 - ramp) * self.initial_rel[leg] + ramp * r
            rel[leg] = r
        return rel


class DLSIKController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, target_speed: float = 0.5):
        self.model = model
        self.data = data
        self.planner = FootstepPlanner(model, data, target_speed)
        self.key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.q_nom = model.key_qpos[self.key].copy()
        self.ctrl_nom = model.key_ctrl[self.key].copy()
        self.q_ik = self.q_nom.copy()
        self.ctrl = self.ctrl_nom.copy()
        self.act_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
        self.jq = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j] for j in range(model.njnt)}
        self.jv = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_dofadr[j] for j in range(model.njnt)}
        self.leg_joints = {
            "left": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
            "right": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
        }
        self.site_id = {
            "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
            "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
        }
        self.range = {name: model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)].copy() for name in self.jq}
        self.ctrl_min = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = model.actuator_ctrlrange[:, 1].copy()
        self.ik_data = mujoco.MjData(model)
        self.last_t = 0.0

    def _solve_leg(self, leg: str, target_world: np.ndarray) -> None:
        cols = np.array([self.jv[j] for j in self.leg_joints[leg]], dtype=int)
        for _ in range(8):
            mujoco.mj_forward(self.model, self.ik_data)
            sid = self.site_id[leg]
            err = target_world - self.ik_data.site_xpos[sid]
            if float(np.linalg.norm(err)) < 1e-4:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, sid)
            J = jacp[:, cols]
            # Damped least squares, solved in task space.
            lam = 2.5e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            dq = np.clip(dq, -0.055, 0.055)
            for joint, delta in zip(self.leg_joints[leg], dq):
                qi = self.jq[joint]
                lo, hi = self.range[joint]
                self.ik_data.qpos[qi] = np.clip(self.ik_data.qpos[qi] + delta, lo + 1e-4, hi - 1e-4)

    def compute(self, data: mujoco.MjData) -> np.ndarray:
        t = float(data.time)
        roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
        rel_feet = self.planner.desired_relative_feet(data, t)
        # IK base: current x/y, gently regulated height, yaw-only orientation.  This
        # is not an applied force; it only defines the inverse-kinematic posture.
        self.ik_data.qpos[:] = self.q_ik
        self.ik_data.qpos[:3] = data.qpos[:3]
        # Slowly bias the target height back to the nominal walking height.
        self.ik_data.qpos[2] = 0.85 * data.qpos[2] + 0.15 * self.planner.nom_height
        _, _, yaw_actual = quat_to_rpy(data.qpos[3:7])
        self.ik_data.qpos[3:7] = yaw_quat(yaw_actual)
        R = yaw_rot(yaw_actual)
        base = self.ik_data.qpos[:3].copy()
        for leg in ("left", "right"):
            target = base + R @ rel_feet[leg]
            self._solve_leg(leg, target)
        self.q_ik[:] = self.ik_data.qpos
        # Convert IK qpos to actuator position targets.  Arms counter-swing.
        ctrl_des = self.ctrl_nom.copy()
        for name, ai in self.act_id.items():
            if name in self.jq:
                ctrl_des[ai] = self.q_ik[self.jq[name]]
        tg, _, _, ramp = self.planner.phase_info(t)
        omega = 2.0 * math.pi / (2.0 * self.planner.step_time)
        arm = 0.30 * ramp * math.sin(omega * tg)
        ctrl_des[self.act_id["left_shoulder_pitch_joint"]] = 0.2 + arm
        ctrl_des[self.act_id["right_shoulder_pitch_joint"]] = 0.2 - arm
        ctrl_des[self.act_id["left_elbow_joint"]] = 1.28 - 0.08 * abs(arm)
        ctrl_des[self.act_id["right_elbow_joint"]] = 1.28 - 0.08 * abs(arm)
        # Soft posture feedback through existing joint targets.  Coefficients are
        # intentionally small; all physical stiffness/damping remains from XML.
        pitch_fb = np.clip(0.35 * pitch + 0.025 * data.qvel[4], -0.08, 0.08)
        roll_fb = np.clip(0.30 * roll + 0.020 * data.qvel[3], -0.07, 0.07)
        for pref in ("left", "right"):
            ctrl_des[self.act_id[f"{pref}_hip_pitch_joint"]] += pitch_fb
            ctrl_des[self.act_id[f"{pref}_ankle_pitch_joint"]] -= 0.35 * pitch_fb
            ctrl_des[self.act_id[f"{pref}_hip_roll_joint"]] += roll_fb
            ctrl_des[self.act_id[f"{pref}_ankle_roll_joint"]] -= 0.35 * roll_fb
        ctrl_des[self.act_id["waist_pitch_joint"]] = np.clip(-0.25 * pitch, -0.10, 0.10)
        ctrl_des[self.act_id["waist_roll_joint"]] = np.clip(-0.25 * roll, -0.10, 0.10)
        ctrl_des = np.clip(ctrl_des, self.ctrl_min, self.ctrl_max)
        # Low-pass motor targets to avoid hard kinematic snapping.
        tau = 0.028
        alpha = min(1.0, self.model.opt.timestep / tau)
        self.ctrl += alpha * (ctrl_des - self.ctrl)
        return self.ctrl


def run(duration: float = 10.0, model_path: str = MODEL, video_path: str | None = None, width: int = 640, height: int = 480, fps: int = 30, target_speed: float = 0.5) -> Dict:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key)
    ctrl = DLSIKController(model, data, target_speed=target_speed)
    renderer = None
    writer = None
    if video_path:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=height, width=width)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.3
    cam.azimuth = 150
    cam.elevation = -14
    cam.lookat[:] = [0.7, 0.0, 0.75]
    start_x = float(data.qpos[0])
    min_z = 9.0
    max_tilt = 0.0
    max_y = 0.0
    fallen = False
    frame_period = 1.0 / fps
    next_frame = 0.0
    samples = []
    steps = int(duration / model.opt.timestep)
    for step in range(steps):
        data.ctrl[:] = ctrl.compute(data)
        mujoco.mj_step(model, data)
        roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
        tilt = max(abs(roll), abs(pitch))
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, float(data.qpos[2]))
        max_y = max(max_y, abs(float(data.qpos[1])))
        if data.qpos[2] < 0.48 or tilt > 1.0:
            fallen = True
        if step % max(1, int(0.5 / model.opt.timestep)) == 0:
            samples.append({"t": float(data.time), "x": float(data.qpos[0]), "y": float(data.qpos[1]), "z": float(data.qpos[2]), "vx": float(data.qvel[0]), "roll": roll, "pitch": pitch, "contacts": int(data.ncon)})
        if renderer is not None and data.time + 1e-9 >= next_frame:
            cam.lookat[:] = [float(data.qpos[0]) + 0.4, float(data.qpos[1]), 0.72]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            next_frame += frame_period
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()
    dist = float(data.qpos[0] - start_x)
    metrics = {
        "duration_s": float(data.time),
        "distance_m": dist,
        "average_speed_mps": dist / float(data.time),
        "target_speed_mps": target_speed,
        "final_x_m": float(data.qpos[0]),
        "final_y_m": float(data.qpos[1]),
        "final_z_m": float(data.qpos[2]),
        "min_base_z_m": min_z,
        "max_abs_y_m": max_y,
        "max_tilt_rad": max_tilt,
        "fallen": fallen,
        "video_path": video_path,
        "video_width": width if video_path else None,
        "video_height": height if video_path else None,
        "video_fps": fps if video_path else None,
        "samples": samples,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--video", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target-speed", type=float, default=0.5)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()
    metrics = run(args.duration, args.model, args.video, args.width, args.height, args.fps, args.target_speed)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.metrics:
        Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics).write_text(text + "\n")


if __name__ == "__main__":
    main()
