#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from g1_dcm_march import Cfg, Marcher, quat_to_rpy, DEFAULT_MODEL


def record(cfg: Cfg, duration: float, video: str, metrics: str, width=640, height=480, fps=30):
    model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL)
    data = mujoco.MjData(model)
    ctrl = Marcher(model, data, cfg)
    renderer = mujoco.Renderer(model, height=height, width=width)
    Path(video).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video, fps=fps, codec='libx264', quality=8, macro_block_size=1)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance = 2.35; cam.azimuth = 145; cam.elevation = -15; cam.lookat[:] = [0.0, 0.0, 0.7]
    dt = model.opt.timestep; next_frame = 0.0
    minz = 9.0; max_tilt = 0.0; fall_time = None; samples=[]; both=sgl=0
    startx=float(data.qpos[0])
    for i in range(int(duration/dt)):
        t=float(data.time)
        data.ctrl[:] = ctrl.compute(data, t)
        mujoco.mj_step(model, data)
        r,p,_=quat_to_rpy(data.qpos[3:7]); tilt=max(abs(r),abs(p)); max_tilt=max(max_tilt,tilt); minz=min(minz,float(data.qpos[2]))
        if t >= ctrl.gait_start:
            cL=ctrl.stance_in_contact(data,'left'); cR=ctrl.stance_in_contact(data,'right')
            if cL and cR: both+=1
            elif cL != cR: sgl+=1
        if (data.qpos[2] < 0.45 or tilt > 1.0) and fall_time is None:
            fall_time=float(data.time)
        if i % max(1,int(0.5/dt)) == 0:
            mujoco.mj_subtreeVel(model,data); com=data.subtree_com[ctrl.pid]
            samples.append(dict(t=round(t,3),x=round(float(data.qpos[0]),3),z=round(float(data.qpos[2]),3),com_y=round(float(com[1]),4),tilt=round(float(tilt),4),ncon=int(data.ncon)))
        if t + 1e-12 >= next_frame:
            cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.70]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            next_frame += 1.0/fps
    writer.close(); renderer.close()
    walk=both+sgl; dist=float(data.qpos[0]-startx)
    out=dict(cfg=cfg.__dict__, duration_s=round(float(data.time),3), distance_m=round(dist,4), avg_speed_mps=round(dist/float(data.time),4), min_base_z=round(minz,4), max_tilt=round(max_tilt,4), fall_time=None if fall_time is None else round(fall_time,3), single_support_frac=round(sgl/walk,3) if walk else 0, double_support_frac=round(both/walk,3) if walk else 0, qfrc_norm=float(np.linalg.norm(data.qfrc_applied)), xfrc_norm=float(np.linalg.norm(data.xfrc_applied)), video=video, video_size=[width,height], fps=fps, samples=samples)
    Path(metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics).write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--duration',type=float,default=10.0); ap.add_argument('--video',default='artifacts/g1_dcm_march_10s_640x480.mp4'); ap.add_argument('--metrics',default='artifacts/g1_dcm_march_metrics.json')
    # Verified config #1
    cfg = Cfg(T=0.38, d=0.115, pelvis_height=0.74, clearance=0.03, init_shift=1.0, init_hold=0.3, init_frac=0.40, sway_frac=0.35, ds_frac=0.35, dcm_gain=1.0, dcm_clip=0.05, min_abs_y=0.06)
    args=ap.parse_args(); record(cfg,args.duration,args.video,args.metrics)
