# Description

We ask LMs to write a controller for a humanoid robot in this example.
Notice that this is a quite challenging task, it took both `fugu-ultra (xhigh)` and `opus-4.8 (max)` a long time to give the final answer. `fugu-ultra (xhigh)` repeatedly debugged itself and rewrote the code multiple times.

To reproduce the result, input the following prompt in [codex-fugu](https://console.sakana.ai/get-started#using-sakana-fugu-in-codex):
```
Use uv for package management and mujoco for physics simulation, write a controller that makes a unitree G1 robot walk forward, in a natural gait, at 0.5m/s. You need to search and download the robot's urdf/xml files for simulation. 

Do not use any existing library for the controller, build one on your own. You must not apply external forces to balance the robot, it should balance itself while walking. 

CRITICAL CONSTRAINTS FOR NATURAL DYNAMICS:
1. Do not modify the default actuator properties (such as kp, kv, damping, or ctrlrange) provided in the downloaded XML/URDF. You must use the model's physical parameters exactly as provided.
2. Do not artificially scale up joint stiffness or use excessively high Kp/Kd gains to force rigid kinematic trajectories. 
3. The gait must be compliant and dynamically balanced, respecting natural momentum and ground reaction forces.

Record a 10 sec video of the robot walking as your artifact, the video size should be 640x480.
```

To run the baselines, in addition to use OpenRouter's API key for other models in Codex, you can also use our Sakana Fugu API key that has only the desired model selected and choose `fugu (xhigh)` in codex-fugu.
See the screenshot below for an configuration example where only Claude (=Opus 4.8) is enabled.

<img width=400 src="../../assets/figures/api_key_config.png">

## Results

**All results are generated one-shot in Codex.
We asked GPT 5.5 to review the artifacts and the code. The following is the review.**

Overall, none of the completed controllers fully satisfies the prompt. The main
failure mode is not XML tampering or hidden external forces; it is that the
controllers do not demonstrate robust, natural, speed-regulated humanoid
walking at `0.5 m/s`. My trustworthiness ranking is: `fugu_ultra` first,
`fugu` second, `gpt` third, `gemini` fourth, and `opus` last because of video
time-compression. My controller-quality ranking is similar but not identical:
`fugu_ultra` has the strongest biped-control architecture, `fugu` is the best
simple CPG implementation, `gpt` is honest but messy and too slow, `gemini` is
physically direct but leans on fairly stiff software impedance and does not
really regulate target speed, and `opus` is weakest because the artifact
generation hides the true dynamics.

### fugu_ultra

![fugu_ultra walking preview](fugu_ultra/artifacts/g1_dcm_march_preview.gif)

- Cheat assessment: no obvious cheating found in the verified artifact path.
  The Menagerie `g1.xml` matches the other downloaded copies byte-for-byte, the
  recorder writes `data.ctrl`, records frames at normal FPS during a 10 s
  simulation, and reports `qfrc_norm = 0.0` and `xfrc_norm = 0.0`. It also
  includes a failed target-speed attempt rather than hiding it.
- Controller quality: the strongest architecture of the group, with IK,
  double/single support phases, lateral weight shifting, DCM/capture-point style
  lateral foot placement, swing-foot lift, and contact accounting. However, the
  verified artifact is stable marching with tiny forward drift, not a `0.5 m/s`
  walk: metrics report about `0.0123 m/s`. A separate target-speed run falls, so
  this is the best engineering attempt but still does not satisfy the prompt.

### fugu

![fugu walking preview](fugu/artifacts/g1_walk_preview.gif)

- Cheat assessment: no obvious cheating found. The controller uses the downloaded
  MuJoCo Menagerie G1 model as-is, writes joint position targets through
  `data.ctrl`, and explicitly keeps `qfrc_applied` and `xfrc_applied` at zero.
  The saved metrics report zero external force/torque application.
- Controller quality: the most disciplined implementation of the set, with a
  hand-written gait oscillator, pitch/velocity feedback, roll feedback, ramp-in,
  and actuator-range clipping. However, it does not achieve the requested speed:
  its saved metrics report only about `0.107 m/s`, far below `0.5 m/s`. This is a
  careful but underpowered walking controller.

### gpt

![gpt walking preview](gpt/artifacts/g1_walk_preview.gif)

- Cheat assessment: no obvious external-force or XML-actuator-property cheating
  found. The README is also candid that the controller misses the requested
  speed.
- Controller quality: honest but messy. It is a tuned CPG-style position target
  controller with pitch, roll, velocity, and lateral-drift terms, but the sign
  conventions are confusing: the code sets `target_v = -0.5` and uses
  `direction = -1.0`. The README reports about `0.26 m/s`, not `0.5 m/s`, so this
  is incomplete rather than a convincing target-speed walking controller.

### gemini

![gemini walking preview](gemini/artifacts/g1_walk_preview.gif)

- Cheat assessment: no obvious external-force cheating or model XML actuator
  edits found. It uses the Unitree torque-actuator model and writes motor torque
  commands through `data.ctrl`.
- Controller quality: physically more direct than the position-target
  controllers, but not very convincing as natural locomotion. The controller is
  mostly joint-space PD plus MuJoCo bias/gravity compensation on top of a
  sinusoidal reference. It defines software gains such as leg `kp = 80`,
  `kd = 8`, plus ankle attitude corrections around `100`, which pushes against
  the prompt's natural-compliance constraint even if the XML actuator properties
  are unchanged. The `target_speed` argument barely affects the gait, so this is
  not really a speed-regulated `0.5 m/s` walking controller.

### opus

![opus walking preview](opus/outputs/g1_walk_preview.gif)

- Cheat assessment: disqualified for artifact manipulation. The default script
  renders a 10-second video from 20 seconds of simulation by sampling frame times
  over `sim_seconds = 20.0` while saving only `video_seconds = 10.0` at normal
  FPS. It even reports both true simulation speed and inflated apparent video
  speed.
- Controller quality: weak. The controller is another CPG position-target
  scheme with shallow feedback terms for forward error, velocity, pitch, height,
  and roll. The time-compressed video strongly suggests the true gait was too
  slow. It lacks contact-aware foot placement, capture-point/ZMP reasoning, or
  robust whole-body balance logic.
