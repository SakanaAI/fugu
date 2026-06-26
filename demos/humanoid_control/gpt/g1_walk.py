from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
OUT = ROOT / "artifacts" / "g1_walk_10s.mp4"


@dataclass(frozen=True)
class GaitParams:
    # The velocity target is intentionally left at -0.5 m/s; diagnostics show
    # the hand-built controller currently underachieves it.
    target_v: float = -0.5
    period: float = 0.4736020252799555
    stance: float = 0.543025316600372
    phase0: float = 0.07071982291881972
    direction: float = -1.0
    base_hp: float = 0.046247827786619476
    base_ap: float = -0.1720546459653843
    base_k: float = 0.20197065934678476
    amp: float = 0.07216088613869795
    lift: float = 0.5006878003101809
    stance_k: float = -0.04764771428196768
    toe: float = 0.12339049937034821
    swing_toe: float = -0.02900491761630482
    ankle_couple: float = 0.7493931599780297
    hip_fb: float = 0.2873514617120188
    ankle_fb: float = -0.18333380294097373
    pitch_k: float = -0.08764058167094574
    pitch_d: float = 0.15482038829016936
    vel_k: float = 0.701450476347253
    lean: float = -0.044285879632931034
    fb_lim: float = 0.18
    lat_amp: float = -0.02326539559642627
    base_roll: float = -0.020316205023654076
    hip_roll_fb: float = 0.09105732344117728
    ankle_roll_fb: float = 0.46173343239409576
    roll_k: float = 0.5615773584917804
    roll_d: float = -0.013871853028167673
    y_k: float = 0.7689684544995468
    hip_yaw: float = 0.01608815283888091
    arm_amp: float = 0.2687669246905989
    waist_yaw: float = 0.05859391840548536
    waist_pitch_fb: float = -0.1323836822272269
    start: float = 0.15
    ramp: float = 0.65


class G1Walker:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if self.key_id < 0:
            raise RuntimeError("Expected the Menagerie G1 scene.xml to contain keyframe 'stand'.")

        self.actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(model.nu)
        ]
        self.act = {name: i for i, name in enumerate(self.actuator_names)}
        self.ctrl_min = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = model.actuator_ctrlrange[:, 1].copy()
        self.base_ctrl = model.key_ctrl[self.key_id].copy()
        self.params = GaitParams()
        self.filtered_vx = 0.0

    @staticmethod
    def pitch_roll(qwxyz: np.ndarray) -> tuple[float, float]:
        w, x, y, z = qwxyz
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return pitch, roll

    def reset(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetDataKeyframe(self.model, data, self.key_id)
        self.filtered_vx = 0.0

    def control(self, data: mujoco.MjData) -> np.ndarray:
        p = self.params
        t = data.time
        pitch, roll = self.pitch_roll(data.qpos[3:7])
        self.filtered_vx = 0.98 * self.filtered_vx + 0.02 * data.qvel[0]

        phase = (t / p.period + p.phase0) % 1.0
        ramp = min(1.0, max(0.0, (t - p.start) / p.ramp))
        ctrl = self.base_ctrl.copy()

        v_err = p.target_v - self.filtered_vx
        fb_pitch = (
            -p.pitch_k * pitch
            - p.pitch_d * data.qvel[4]
            - p.vel_k * v_err
            + p.lean
        )
        fb_pitch = float(np.clip(fb_pitch, -p.fb_lim, p.fb_lim))

        lateral = (
            p.lat_amp * math.sin(2.0 * math.pi * phase)
            - p.roll_k * roll
            - p.roll_d * data.qvel[3]
            - p.y_k * data.qpos[1]
        )

        for side, offset, sign in (("left", 0.0, 1.0), ("right", 0.5, -1.0)):
            leg_phase = (phase + offset) % 1.0
            if leg_phase < p.stance:
                s = leg_phase / p.stance
                stride = p.direction * (-1.0 + 2.0 * s)
                knee = p.base_k + p.stance_k * math.sin(math.pi * s)
                toe = p.toe * max(0.0, (s - 0.55) / 0.45) ** 2
            else:
                s = (leg_phase - p.stance) / (1.0 - p.stance)
                stride = p.direction * (1.0 - 2.0 * s)
                knee = p.base_k + p.lift * math.sin(math.pi * s)
                toe = -p.swing_toe * math.sin(math.pi * s)

            hip_pitch = p.base_hp + p.amp * stride + p.hip_fb * fb_pitch
            ankle_pitch = (
                p.base_ap
                - p.ankle_couple * p.amp * stride
                + p.ankle_fb * fb_pitch
                + toe
            )
            hip_roll = sign * (p.base_roll + p.hip_roll_fb * lateral)
            ankle_roll = -sign * (p.ankle_roll_fb * lateral)
            hip_yaw = sign * p.hip_yaw * math.sin(2.0 * math.pi * leg_phase)

            ctrl[self.act[f"{side}_hip_pitch_joint"]] = hip_pitch * ramp
            ctrl[self.act[f"{side}_knee_joint"]] = knee * ramp
            ctrl[self.act[f"{side}_ankle_pitch_joint"]] = ankle_pitch * ramp
            ctrl[self.act[f"{side}_hip_roll_joint"]] = hip_roll * ramp
            ctrl[self.act[f"{side}_ankle_roll_joint"]] = ankle_roll * ramp
            ctrl[self.act[f"{side}_hip_yaw_joint"]] = hip_yaw * ramp

        arm = p.arm_amp * math.sin(2.0 * math.pi * phase) * ramp
        ctrl[self.act["left_shoulder_pitch_joint"]] = (
            self.base_ctrl[self.act["left_shoulder_pitch_joint"]] - arm
        )
        ctrl[self.act["right_shoulder_pitch_joint"]] = (
            self.base_ctrl[self.act["right_shoulder_pitch_joint"]] + arm
        )
        ctrl[self.act["waist_yaw_joint"]] = p.waist_yaw * math.sin(2.0 * math.pi * phase) * ramp
        ctrl[self.act["waist_pitch_joint"]] = p.waist_pitch_fb * fb_pitch * ramp

        return np.clip(ctrl, self.ctrl_min, self.ctrl_max)


def simulate(render: bool, duration: float, width: int, height: int, fps: int, output: Path) -> dict[str, float]:
    if not MODEL.exists():
        raise FileNotFoundError(
            f"{MODEL} is missing. Download the model with the repository sparse clone commands in README.md."
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    walker = G1Walker(model)
    walker.reset(data)

    renderer = mujoco.Renderer(model, height=height, width=width) if render else None
    camera = mujoco.MjvCamera() if render else None
    if camera is not None:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 135.0
        camera.elevation = -18.0
        camera.distance = 2.4
    writer = None
    if render:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)

    next_frame = 0.0
    start_x = float(data.qpos[0])
    max_pitch = 0.0
    max_roll = 0.0
    max_abs_y = 0.0
    min_height = float(data.qpos[2])
    max_ctrl_norm = 0.0

    try:
        while data.time < duration - 1e-12:
            ctrl = walker.control(data)
            data.ctrl[:] = ctrl
            max_ctrl_norm = max(max_ctrl_norm, float(np.linalg.norm(ctrl)))
            mujoco.mj_step(model, data)

            pitch, roll = walker.pitch_roll(data.qpos[3:7])
            max_pitch = max(max_pitch, abs(pitch))
            max_roll = max(max_roll, abs(roll))
            max_abs_y = max(max_abs_y, abs(float(data.qpos[1])))
            min_height = min(min_height, float(data.qpos[2]))

            if render and data.time + 1e-12 >= next_frame:
                assert renderer is not None and writer is not None and camera is not None
                camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.35])
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                next_frame += 1.0 / fps
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    distance = float(data.qpos[0] - start_x)
    return {
        "duration_s": float(data.time),
        "distance_x_m": distance,
        "mean_vx_mps": distance / float(data.time),
        "target_vx_mps": walker.params.target_v,
        "min_pelvis_height_m": min_height,
        "max_abs_lateral_y_m": max_abs_y,
        "max_abs_pitch_rad": max_pitch,
        "max_abs_roll_rad": max_roll,
        "max_ctrl_l2": max_ctrl_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    metrics = simulate(
        render=not args.no_render,
        duration=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output=args.output,
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    if not args.no_render:
        print(f"video: {args.output}")


if __name__ == "__main__":
    main()
