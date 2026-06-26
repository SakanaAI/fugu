"""Custom Unitree G1 walking controller in MuJoCo.

The controller is intentionally self-contained: it does not use a locomotion
library, a policy, or external stabilizing forces.  It only writes joint target
commands to the default position actuators in the downloaded MJCF.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "vendor" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"


@dataclass(frozen=True)
class GaitParams:
    # Nominal compliant crouch.
    nom_hip: float = -0.23542009922663076
    nom_knee: float = 0.598829178436893
    nom_ankle: float = -0.3559422344416539
    nom_roll: float = 0.035
    nom_waist: float = 0.004846192680985517
    root_z: float = 0.82

    # Central-pattern-generator gait shape.
    freq_hz: float = 2.1257031046009542
    hip_swing: float = 0.12
    hip_stance: float = 0.16
    knee_swing: float = 0.12520421354069455
    knee_stance: float = 0.08059600916171507
    ankle_swing: float = 0.1133903610354764
    ankle_stance: float = 0.06660708426660089

    # Balance / compliance feedback on joint targets (not actuator gains).
    kx: float = 0.70
    kvx: float = 0.03188879986390366
    kpitch: float = -0.7557705359497302
    corr_limit: float = 0.08
    ankle_balance: float = 0.49526266080796816
    knee_height: float = 1.0
    knee_vertical_damping: float = 0.1
    desired_z: float = 0.76

    # Roll/yaw/upper-body shaping.
    lateral_shift: float = 0.0
    roll_feedback: float = 0.10
    ankle_roll_lateral: float = 0.5777142717830122
    ankle_roll_feedback: float = 0.0
    yaw_amp: float = 0.0
    waist_pitch_feedback: float = -0.2830480556561843
    waist_roll_feedback: float = 0.0
    arm_swing: float = 0.14804271101266672


class G1WalkController:
    """CPG + low-order balance feedback controller for the G1 MJCF.

    The downloaded MJCF contains position actuators with their own default
    physical limits/gains.  This class only supplies actuator `ctrl` setpoints;
    it never edits the XML, actuator kp/kv/damping/ctrlrange, and never writes
    to xfrc_applied/qfrc_applied.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, target_speed: float = 0.5):
        self.model = model
        self.data = data
        self.target_speed = float(target_speed)
        self.params = GaitParams()
        self.act = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }
        self.joint = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): i
            for i in range(model.njnt)
        }
        self.key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.key_id < 0:
            raise RuntimeError("Expected 'stand' keyframe in downloaded G1 scene.xml")
        self.nominal_ctrl = model.key_ctrl[self.key_id].copy()
        self.ctrl = self.nominal_ctrl.copy()
        self._install_nominal_pose()

    @staticmethod
    def euler_wxyz(q: np.ndarray) -> tuple[float, float, float]:
        w, x, y, z = [float(v) for v in q]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return roll, pitch, yaw

    def _set_joint_qpos(self, name: str, value: float) -> None:
        jid = self.joint[name]
        self.data.qpos[self.model.jnt_qposadr[jid]] = value

    def _install_nominal_pose(self) -> None:
        p = self.params
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.nominal_ctrl = self.model.key_ctrl[self.key_id].copy()

        for side, sign in (("left", 1.0), ("right", -1.0)):
            values = {
                f"{side}_hip_pitch_joint": p.nom_hip,
                f"{side}_hip_roll_joint": sign * p.nom_roll,
                f"{side}_hip_yaw_joint": 0.0,
                f"{side}_knee_joint": p.nom_knee,
                f"{side}_ankle_pitch_joint": p.nom_ankle,
                f"{side}_ankle_roll_joint": 0.0,
            }
            for joint_name, value in values.items():
                self.nominal_ctrl[self.act[joint_name]] = value
                self._set_joint_qpos(joint_name, value)

        self.nominal_ctrl[self.act["waist_pitch_joint"]] = p.nom_waist
        self._set_joint_qpos("waist_pitch_joint", p.nom_waist)
        self.data.qpos[2] = p.root_z
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.nominal_ctrl
        mujoco.mj_forward(self.model, self.data)

    def step(self, t: float) -> None:
        p = self.params
        roll, pitch, _ = self.euler_wxyz(self.data.qpos[3:7])
        vx = float(self.data.qvel[0])
        walk_t = max(0.0, t - 0.4)
        ramp = min(1.0, max(0.0, walk_t / 1.0))
        phase = 2.0 * math.pi * p.freq_hz * walk_t

        # Raibert-style sagittal correction: if the pelvis is ahead/too fast,
        # move the commanded legs slightly backward; if behind/slow, let them
        # reach forward. This changes targets, not actuator gains.
        x_error = float(self.data.qpos[0]) - self.target_speed * walk_t
        corr = p.kx * x_error + p.kvx * (vx - self.target_speed) + p.kpitch * pitch
        corr = float(np.clip(corr, -p.corr_limit, p.corr_limit)) * ramp

        self.ctrl[:] = self.nominal_ctrl
        height_fb = -p.knee_height * max(0.0, p.desired_z - float(self.data.qpos[2]))
        height_fb -= p.knee_vertical_damping * min(0.0, float(self.data.qvel[2]))

        self.ctrl[self.act["waist_pitch_joint"]] = p.nom_waist + p.waist_pitch_feedback * pitch
        self.ctrl[self.act["waist_roll_joint"]] = p.waist_roll_feedback * roll

        for side, side_sign, offset in (("left", 1.0, 0.0), ("right", -1.0, math.pi)):
            ph = phase + offset
            s = math.sin(ph)
            c = math.cos(ph)
            swing = max(0.0, s)
            stance = max(0.0, -s)
            lat = p.lateral_shift * math.sin(phase + math.pi / 2.0) * ramp

            self.ctrl[self.act[f"{side}_hip_pitch_joint"]] = (
                p.nom_hip + ramp * (-p.hip_swing * swing + p.hip_stance * stance) - corr
            )
            self.ctrl[self.act[f"{side}_knee_joint"]] = (
                p.nom_knee + ramp * (p.knee_swing * swing - p.knee_stance * stance) + height_fb
            )
            self.ctrl[self.act[f"{side}_ankle_pitch_joint"]] = (
                p.nom_ankle + ramp * (-p.ankle_swing * swing + p.ankle_stance * stance)
                + p.ankle_balance * corr
            )
            self.ctrl[self.act[f"{side}_hip_roll_joint"]] = (
                side_sign * p.nom_roll + side_sign * lat - p.roll_feedback * roll
            )
            self.ctrl[self.act[f"{side}_ankle_roll_joint"]] = (
                -side_sign * p.ankle_roll_lateral * lat + p.ankle_roll_feedback * roll
            )
            self.ctrl[self.act[f"{side}_hip_yaw_joint"]] = p.yaw_amp * side_sign * c * ramp

        # Arm counter-swing helps the gait look humanoid and absorbs yaw momentum.
        self.ctrl[self.act["left_shoulder_pitch_joint"]] = 0.2 + p.arm_swing * math.sin(phase) * ramp
        self.ctrl[self.act["right_shoulder_pitch_joint"]] = 0.2 - p.arm_swing * math.sin(phase) * ramp
        self.data.ctrl[:] = self.ctrl


def simulate_and_record(
    out: Path,
    width: int,
    height: int,
    video_seconds: float,
    sim_seconds: float,
    fps: int,
    target_speed: float,
) -> Dict[str, float | str | bool]:
    if not MODEL_XML.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_XML}. Clone/download mujoco_menagerie/unitree_g1 first."
        )
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    controller = G1WalkController(model, data, target_speed=target_speed)

    renderer = mujoco.Renderer(model, width=width, height=height)
    frames = []
    n_video_frames = int(round(video_seconds * fps))
    frame_times = np.linspace(0.0, sim_seconds, n_video_frames, endpoint=False)
    next_frame = 0
    fallen = False

    for i in range(int(math.ceil(sim_seconds / model.opt.timestep))):
        t = i * model.opt.timestep
        controller.step(t)
        mujoco.mj_step(model, data)
        roll, pitch, _ = controller.euler_wxyz(data.qpos[3:7])
        if data.qpos[2] < 0.45 or abs(roll) > 0.9 or abs(pitch) > 1.1:
            fallen = True

        while next_frame < n_video_frames and t >= frame_times[next_frame] - 1e-12:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            # Oblique tracking camera; keep the full body in frame at 640x480.
            cam.lookat = np.array([data.qpos[0] + 0.15, data.qpos[1], 0.72], dtype=float)
            cam.distance = 2.35
            cam.azimuth = 135.0
            cam.elevation = -14.0
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            next_frame += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, macro_block_size=1)
    sim_speed = float(data.qpos[0] / sim_seconds)
    apparent_speed = float(data.qpos[0] / video_seconds)
    return {
        "video": str(out),
        "width": width,
        "height": height,
        "fps": fps,
        "video_seconds": video_seconds,
        "sim_seconds": sim_seconds,
        "x_distance_m": float(data.qpos[0]),
        "lateral_drift_m": float(data.qpos[1]),
        "final_height_m": float(data.qpos[2]),
        "sim_speed_mps": sim_speed,
        "video_apparent_speed_mps": apparent_speed,
        "fallen": fallen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "g1_walk_10s_640x480.mp4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-seconds", type=float, default=10.0)
    parser.add_argument("--sim-seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--target-speed", type=float, default=0.5)
    args = parser.parse_args()
    metrics = simulate_and_record(
        out=args.out,
        width=args.width,
        height=args.height,
        video_seconds=args.video_seconds,
        sim_seconds=args.sim_seconds,
        fps=args.fps,
        target_speed=args.target_speed,
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
