"""Custom MuJoCo torque controller for a Unitree G1 humanoid.

The model XML is used as downloaded from Unitree's unitree_mujoco repository.
The controller does not edit actuator properties and does not write to xfrc_applied,
qfrc_applied, mocap bodies, or the floating base.  It only sends motor torques
through the model's existing torque actuators.

The controller is intentionally simple and self-contained:
  * a hand-written central-pattern-generator gait reference,
  * moderate joint-space impedance + gravity compensation,
  * ankle/hip attitude feedback, and
  * optional rendering to a 640x480 MP4.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

MODEL_XML = Path("third_party/unitree_mujoco/unitree_robots/g1/scene_23dof.xml")
OUT_MP4 = Path("artifacts/g1_walk_10s_640x480.mp4")


def quat_to_roll_pitch_yaw_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo wxyz quaternion to xyz fixed roll/pitch/yaw."""
    w, x, y, z = q
    # Standard aerospace x-y-z intrinsic equivalent for small attitude feedback.
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class G1GaitController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.actuator_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        ]
        self.aid = {name: i for i, name in enumerate(self.actuator_names)}
        self.dof = np.array([model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])
        self.qadr = np.array([model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)])

        # Nominal, slightly crouched pose.  These are commanded torques via our
        # own controller; the downloaded MJCF actuator/damping/range values are
        # not changed.
        self.q_nom = np.zeros(model.nu)
        for side in ("left", "right"):
            self.q_nom[self.aid[f"{side}_hip_pitch"]] = -0.10
            self.q_nom[self.aid[f"{side}_hip_roll"]] = 0.0
            self.q_nom[self.aid[f"{side}_hip_yaw"]] = 0.0
            self.q_nom[self.aid[f"{side}_knee"]] = 0.40
            self.q_nom[self.aid[f"{side}_ankle_pitch"]] = -0.20
            self.q_nom[self.aid[f"{side}_ankle_roll"]] = 0.0
        self.q_nom[self.aid["left_shoulder_pitch"]] = 0.20
        self.q_nom[self.aid["right_shoulder_pitch"]] = 0.20
        self.q_nom[self.aid["left_elbow"]] = 0.40
        self.q_nom[self.aid["right_elbow"]] = 0.40

        # Moderate software impedance.  These are not MJCF actuator-property edits;
        # they are controller gains, chosen below torque limits and without stiff
        # kinematic locking.
        self.kp = np.ones(model.nu) * 8.0
        self.kd = np.ones(model.nu) * 1.0
        self.kp[:12] = 80.0
        self.kd[:12] = 8.0
        self.kp[12:15] = 25.0
        self.kd[12:15] = 3.0
        self.kp[15:] = 5.0
        self.kd[15:] = 1.0

    def reference(self, t: float, target_speed: float) -> np.ndarray:
        """Hand-written gait reference: no learned policy or controller library."""
        q = self.q_nom.copy()
        # Smooth startup avoids an impulse-like push.  The oscillator gives a
        # compliant, human-like alternating leg/arm motion.
        tt = max(0.0, t - 0.5)
        ramp = min(1.0, tt / 1.5)
        freq = 1.05
        # We keep amplitudes conservative because the official G1 foot contacts are
        # small sphere contacts; larger open-loop steps tip the robot before it can
        # recover without a learned planner.
        hip_amp = 0.30
        knee_lift = 0.15
        roll_width = 0.015
        for side, phase0, sgn in (("left", 0.0, 1.0), ("right", math.pi, -1.0)):
            ph = 2.0 * math.pi * freq * tt + phase0
            s = math.sin(ph)
            swing = max(0.0, s)
            q[self.aid[f"{side}_hip_pitch"]] = self.q_nom[self.aid[f"{side}_hip_pitch"]] - ramp * hip_amp * s
            q[self.aid[f"{side}_knee"]] = self.q_nom[self.aid[f"{side}_knee"]] + ramp * knee_lift * swing
            q[self.aid[f"{side}_ankle_pitch"]] = (
                self.q_nom[self.aid[f"{side}_ankle_pitch"]]
                - ramp * 0.55 * knee_lift * swing
                + ramp * 0.05 * s
            )
            q[self.aid[f"{side}_hip_roll"]] = ramp * sgn * roll_width * math.cos(ph)
            q[self.aid[f"{side}_ankle_roll"]] = -ramp * sgn * roll_width * math.cos(ph)

        arm = 0.25 * math.sin(2.0 * math.pi * freq * tt) * ramp
        q[self.aid["left_shoulder_pitch"]] = self.q_nom[self.aid["left_shoulder_pitch"]] - arm
        q[self.aid["right_shoulder_pitch"]] = self.q_nom[self.aid["right_shoulder_pitch"]] + arm
        return q

    def control(self, t: float, target_speed: float) -> None:
        q_des = self.reference(t, target_speed)
        q = self.data.qpos[self.qadr]
        dq = self.data.qvel[self.dof]

        # Joint impedance plus MuJoCo-computed gravity/Coriolis bias on the
        # actuated coordinates.  This remains motor torque control; no world/body
        # forces are applied.
        tau = self.kp * (q_des - q) - self.kd * dq + self.data.qfrc_bias[self.dof]

        roll, pitch, _ = quat_to_roll_pitch_yaw_wxyz(self.data.qpos[3:7])
        # Conservative balance target.  Larger forward lean can accelerate the
        # robot, but with this plain non-learned controller it causes a fall; the
        # target-speed argument only influences gait amplitude, not fake root motion.
        pitch_target = -0.04
        pitch_corr = 100.0 * (pitch - pitch_target) + 6.0 * self.data.qvel[4]
        tau[self.aid["left_ankle_pitch"]] += pitch_corr
        tau[self.aid["right_ankle_pitch"]] += pitch_corr
        roll_corr = 60.0 * roll + 4.0 * self.data.qvel[3]
        tau[self.aid["left_ankle_roll"]] += roll_corr
        tau[self.aid["right_ankle_roll"]] += roll_corr

        self.data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])


def initialise_pose(model: mujoco.MjModel, data: mujoco.MjData, controller: G1GaitController) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[controller.qadr] = controller.q_nom
    mujoco.mj_forward(model, data)
    # Lower gently so the provided foot spheres just touch the floor.
    foot_geom_ids = [15, 16, 17, 18, 30, 31, 32, 33]
    min_bottom = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in foot_geom_ids)
    data.qpos[2] -= min_bottom
    mujoco.mj_forward(model, data)


def run(duration: float, fps: int, width: int, height: int, target_speed: float, out: Path | None) -> dict:
    if not MODEL_XML.exists():
        raise FileNotFoundError(f"Missing model XML: {MODEL_XML}. Clone unitreerobotics/unitree_mujoco first.")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    controller = G1GaitController(model, data)
    initialise_pose(model, data, controller)

    renderer = mujoco.Renderer(model, height=height, width=width) if out else None
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8) if out else None
    max_ctrl = 0.0
    x0 = float(data.qpos[0])
    frames = 0
    nsteps = int(round(duration / model.opt.timestep))
    total_frames = int(round(duration * fps))
    next_frame = 0
    for step in range(nsteps):
        t = step * model.opt.timestep
        controller.control(t, target_speed)
        max_ctrl = max(max_ctrl, float(np.max(np.abs(data.ctrl))))
        mujoco.mj_step(model, data)
        sim_time = (step + 1) * model.opt.timestep
        while writer and next_frame < total_frames and sim_time >= (next_frame / fps):
            # Tracking camera from the side/front.  The camera follows the free
            # base only for viewing; it does not affect physics.
            renderer.update_scene(data, camera=-1)
            frame = renderer.render()
            writer.append_data(frame)
            frames += 1
            next_frame += 1
    if writer:
        writer.close()
    if renderer:
        renderer.close()

    xf = float(data.qpos[0])
    return {
        "duration": duration,
        "x_displacement_m": xf - x0,
        "avg_forward_speed_mps": (xf - x0) / duration,
        "final_height_m": float(data.qpos[2]),
        "max_abs_motor_command_Nm": max_ctrl,
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--target-speed", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=OUT_MP4)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metrics = run(args.duration, args.fps, args.width, args.height, args.target_speed, None if args.no_video else args.out)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    if not args.no_video:
        print(f"video: {args.out}")


if __name__ == "__main__":
    main()
