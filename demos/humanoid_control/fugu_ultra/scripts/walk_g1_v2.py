"""Unitree G1 reactive walking controller for MuJoCo (custom, no control libs).

Design (built from scratch, no locomotion/control library):
  * Model: MuJoCo Menagerie unitree_g1/scene.xml, used verbatim. The position
    actuators (kp=500, dampratio=1) and all joint damping/armature/ctrlrange are
    the model's own defaults -- we never edit them.
  * Balance: Linear Inverted Pendulum (LIPM) + Raibert/capture-point reactive
    foot placement. The swing foot is placed using the MEASURED CoM velocity, so
    the robot catches itself -- this is what keeps it upright, not stiff tracking.
  * Stepping: a gait clock alternates stance/swing legs; the swing foot follows a
    smooth cycloid in (x,z); foot targets are turned into joint position targets
    by a 6-DOF damped-least-squares leg IK (position + flat-sole orientation).
  * The ONLY thing written to the simulator is data.ctrl for the existing
    actuators. We never touch qfrc_applied / xfrc_applied (no external forces).
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import mujoco

DEFAULT_MODEL = "third_party/mujoco_menagerie/unitree_g1/scene.xml"


def smoothstep(x):
    x = min(1.0, max(0.0, x)); return x*x*(3-2*x)

def quat_to_rpy(q):
    w,x,y,z=q
    return (math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            math.asin(max(-1,min(1,2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


class LegIK:
    """6-DOF damped-least-squares IK for both legs (pos + flat-sole)."""
    def __init__(self, model):
        self.m = model
        self.ik = mujoco.MjData(model)
        JN={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j):j for j in range(model.njnt)}
        self.qadr={n:model.jnt_qposadr[JN[n]] for n in JN}
        self.dadr={n:model.jnt_dofadr[JN[n]] for n in JN}
        self.jr={n:model.jnt_range[JN[n]].copy() for n in JN}
        self.LEGS={'left':['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint',
                           'left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint'],
                   'right':['right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint',
                            'right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint']}
        self.SITE={'left':mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'left_foot'),
                   'right':mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'right_foot')}
        self.Rt=np.eye(3)

    def solve(self, base_pos, base_quat, foot_tgt, seed, iters=60):
        ik=self.ik; m=self.m
        ik.qpos[:]=seed; ik.qpos[:3]=base_pos; ik.qpos[3:7]=base_quat
        # Break the straight-leg singularity: ensure a minimum knee bend in the
        # seed so the position Jacobian is well conditioned for vertical motion.
        for leg in ('left','right'):
            kq=self.qadr[f'{leg}_knee_joint']
            if ik.qpos[kq] < 0.2:
                ik.qpos[kq]=0.3
        for it in range(iters):
            mujoco.mj_kinematics(m,ik); mujoco.mj_comPos(m,ik)
            worst=0.0
            for leg in ('left','right'):
                cols=np.array([self.dadr[j] for j in self.LEGS[leg]]); sid=self.SITE[leg]
                perr=foot_tgt[leg]-ik.site_xpos[sid]
                Rc=ik.site_xmat[sid].reshape(3,3); Re=self.Rt@Rc.T
                rerr=0.5*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])
                worst=max(worst,np.linalg.norm(perr))
                jp=np.zeros((3,m.nv)); jrt=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,ik,jp,jrt,sid)
                J=np.vstack([jp[:,cols],jrt[:,cols]]); e=np.concatenate([perr,rerr])
                dq=J.T@np.linalg.solve(J@J.T+1e-4*np.eye(6),e)*0.8
                dq=np.clip(dq,-0.25,0.25)
                for j,dd in zip(self.LEGS[leg],dq):
                    qi=self.qadr[j]; lo,hi=self.jr[j]
                    ik.qpos[qi]=np.clip(ik.qpos[qi]+dd,lo+1e-3,hi-1e-3)
            if worst<5e-4: break
        return ik.qpos.copy()


class Walker:
    def __init__(self, model, data, p):
        self.m=model; self.p=p
        self.ik=LegIK(model)
        self.key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'stand')
        self.act={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i):i for i in range(model.nu)}
        self.cmin=model.actuator_ctrlrange[:,0].copy(); self.cmax=model.actuator_ctrlrange[:,1].copy()
        self.pid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'pelvis')
        self.qadr=self.ik.qadr
        self.H=p['height']
        self.half=0.12         # nominal foot half-width (y)
        self.footz=0.0331
        # initialize standing crouch via IK and snap robot into it
        seed=model.key_qpos[self.key].copy()
        ft={'left':np.array([0.0,self.half,self.footz]),'right':np.array([0.0,-self.half,self.footz])}
        q0=self.ik.solve(np.array([0,0,self.H]),np.array([1,0,0,0]),ft,seed,iters=200)
        data.qpos[:]=q0; data.qpos[:3]=[0,0,self.H]; data.qpos[3:7]=[1,0,0,0]; data.qvel[:]=0
        mujoco.mj_forward(model,data)
        self.q_seed=q0.copy()
        self.ctrl=model.key_ctrl[self.key].copy()
        for leg in('left','right'):
            for j in self.ik.LEGS[leg]:
                if j in self.act: self.ctrl[self.act[j]]=q0[self.qadr[j]]
        # set arm/waist nominal
        for n,v in [('left_shoulder_pitch_joint',0.2),('left_shoulder_roll_joint',0.18),('left_elbow_joint',1.0),
                    ('right_shoulder_pitch_joint',0.2),('right_shoulder_roll_joint',-0.18),('right_elbow_joint',1.0)]:
            if n in self.act: self.ctrl[self.act[n]]=v
        # gait state
        self.t0=p['start_time']
        self.Ts=p['step_time']
        self.stance='left'    # which foot is on the ground first
        self.step_idx=-1
        # world placement of each foot (filled at stance switches)
        self.foot_world={'left':np.array([0.0,self.half,self.footz]),
                         'right':np.array([0.0,-self.half,self.footz])}
        self.swing_from=self.foot_world['right'].copy()
        self.swing_to=self.foot_world['right'].copy()
        self.phase_t0=self.t0

    def com_state(self, data):
        c=data.subtree_com[self.pid].copy()
        # CoM velocity via cvel of pelvis subtree is messy; use finite diff handled outside
        return c

    def begin_step(self, data, t):
        """Choose next swing foot + target via capture-point foot placement."""
        p=self.p
        self.step_idx+=1
        swing='right' if self.stance=='left' else 'left'
        self.swing=swing
        self.phase_t0=t
        # measured CoM and velocity
        c=data.subtree_com[self.pid].copy()
        v=self.com_vel.copy()
        stance_foot=self.foot_world[self.stance]
        # LIPM natural frequency
        w=math.sqrt(9.81/self.H)
        # capture point (x): where to step so we don't fall; track desired vx
        vx_des=p['vx']
        # Raibert-style: neutral point under CoM advanced by half a step + velocity error feedback
        x_cap = c[0] + v[0]/w
        x_step = x_cap + p['kx']*(v[0]-vx_des) + 0.5*vx_des*self.Ts
        # lateral: step to the swing side relative to CoM, with capture term
        y_cap = c[1] + v[1]/w
        side = self.half if swing=='left' else -self.half
        y_step = c[1] + side + p['ky']*v[1]
        # clamp lateral so feet don't cross / over-abduct
        if swing=='left':  y_step=max(c[1]+0.06, min(c[1]+0.20, y_step))
        else:              y_step=min(c[1]-0.06, max(c[1]-0.20, y_step))
        # clamp step length
        x_step=min(stance_foot[0]+p['max_step'], max(stance_foot[0]-0.10, x_step))
        self.swing_from=self.foot_world[swing].copy()
        self.swing_to=np.array([x_step,y_step,self.footz])

    def compute(self, data, t):
        p=self.p; m=self.m
        ramp=smoothstep((t-self.t0)/p['ramp_time']) if t>=self.t0 else 0.0
        # update gait clock
        if t>=self.t0:
            local=(t-self.phase_t0)
            if self.step_idx<0 or local>=self.Ts:
                # commit swing foot to ground, switch stance
                if self.step_idx>=0:
                    self.foot_world[self.swing]=self.swing_to.copy()
                    self.stance=self.swing
                self.begin_step(data,t)
                local=0.0
            tau=min(1.0, local/self.Ts)
        else:
            tau=0.0
        # desired pelvis: follow CoM forward at vx, keep height H
        base_x=float(data.qpos[0])
        base_y=float(data.qpos[1])
        base_pos=np.array([base_x, base_y, self.H])
        _,_,yaw=quat_to_rpy(data.qpos[3:7])
        base_quat=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
        # foot targets in world
        ft={}
        if t<self.t0:
            ft['left']=self.foot_world['left']; ft['right']=self.foot_world['right']
        else:
            for leg in('left','right'):
                if leg==self.stance:
                    ft[leg]=self.foot_world[leg].copy()
                else:
                    s=smoothstep(tau)
                    x=(1-s)*self.swing_from[0]+s*self.swing_to[0]
                    y=(1-s)*self.swing_from[1]+s*self.swing_to[1]
                    z=self.footz + p['clearance']*math.sin(math.pi*tau)
                    ft[leg]=np.array([x,y,z])
        # IK -> joint targets
        qsol=self.ik.solve(base_pos,base_quat,ft,self.q_seed,iters=p['ik_iters'])
        self.q_seed=qsol.copy()
        des=self.ctrl.copy()
        for leg in('left','right'):
            for j in self.ik.LEGS[leg]:
                if j in self.act:
                    tgt=qsol[self.qadr[j]]
                    nom=self.ctrl[self.act[j]]
                    des[self.act[j]]=(1-ramp)*nom+ramp*tgt
        # arm counter-swing (cosmetic / momentum), small
        phi=(t-self.t0)/(2*self.Ts)
        arm=ramp*p['arm_amp']*math.sin(2*math.pi*phi)
        if 'left_shoulder_pitch_joint' in self.act: des[self.act['left_shoulder_pitch_joint']]=0.2+arm
        if 'right_shoulder_pitch_joint' in self.act: des[self.act['right_shoulder_pitch_joint']]=0.2-arm
        des=np.clip(des,self.cmin,self.cmax)
        return des


def simulate(model_path, p, duration, video_path, metrics_path, width, height, fps, camera_track=True):
    model=mujoco.MjModel.from_xml_path(model_path); data=mujoco.MjData(model)
    w=Walker(model,data,p)
    # CoM velocity estimator (finite diff)
    w.com_vel=np.zeros(3); prev_com=data.subtree_com[w.pid].copy()
    renderer=None; writer=None
    if video_path:
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(model,height=height,width=width)
        writer=imageio.get_writer(video_path,fps=fps,codec="libx264",quality=8,macro_block_size=1)
    cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.distance=3.0; cam.azimuth=130; cam.elevation=-12; cam.lookat[:]=[0,0,0.6]
    minz=9; maxt=0; fallen=False; samples=[]; next_frame=0.0; fp=1.0/fps
    start_x=float(data.qpos[0])
    steps=int(duration/model.opt.timestep)
    for s in range(steps):
        t=float(data.time)
        # update CoM velocity (finite diff, low-pass)
        com=data.subtree_com[w.pid].copy()
        inst=(com-prev_com)/model.opt.timestep; prev_com=com
        w.com_vel=0.92*w.com_vel+0.08*inst
        data.ctrl[:]=w.compute(data,t)
        mujoco.mj_step(model,data)
        r,pi,ya=quat_to_rpy(data.qpos[3:7]); tilt=max(abs(r),abs(pi))
        minz=min(minz,float(data.qpos[2])); maxt=max(maxt,tilt)
        if data.qpos[2]<0.45 or tilt>1.0: fallen=True
        if s%int(0.5/model.opt.timestep)==0:
            samples.append({"t":round(t,3),"x":round(float(data.qpos[0]),3),"y":round(float(data.qpos[1]),3),
                            "z":round(float(data.qpos[2]),3),"vx":round(float(w.com_vel[0]),3),
                            "roll":round(r,3),"pitch":round(pi,3),"ncon":int(data.ncon)})
        if renderer is not None and t+1e-12>=next_frame:
            if camera_track: cam.lookat[:]=[float(data.qpos[0]),float(data.qpos[1]),0.5]
            renderer.update_scene(data,camera=cam); writer.append_data(renderer.render()); next_frame+=fp
    if writer: writer.close()
    if renderer: renderer.close()
    dist=float(data.qpos[0]-start_x)
    res={"model_path":model_path,"params":p,"duration_s":round(float(data.time),3),
         "distance_m":round(dist,3),"average_speed_mps":round(dist/float(data.time),4),
         "target_speed_mps":p['vx'],"final_xyz":[round(float(x),3) for x in data.qpos[:3]],
         "min_base_z_m":round(minz,3),"max_tilt_rad":round(maxt,3),"fallen":fallen,
         "video":video_path,"video_size":[width,height] if video_path else None,"fps":fps if video_path else None,
         "xfrc_applied_norm":float(np.linalg.norm(data.xfrc_applied)),
         "qfrc_applied_norm":float(np.linalg.norm(data.qfrc_applied)),
         "samples":samples}
    txt=json.dumps(res,indent=2); print(txt)
    if metrics_path: Path(metrics_path).parent.mkdir(parents=True,exist_ok=True); Path(metrics_path).write_text(txt+"\n")
    return res


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--duration",type=float,default=10.0)
    ap.add_argument("--video",default=None)
    ap.add_argument("--metrics",default=None)
    ap.add_argument("--width",type=int,default=640); ap.add_argument("--height",type=int,default=480)
    ap.add_argument("--fps",type=int,default=30)
    ap.add_argument("--vx",type=float,default=0.5)
    ap.add_argument("--height-com",type=float,default=0.70)
    ap.add_argument("--step-time",type=float,default=0.40)
    ap.add_argument("--clearance",type=float,default=0.06)
    ap.add_argument("--kx",type=float,default=0.10)
    ap.add_argument("--ky",type=float,default=0.20)
    ap.add_argument("--max-step",type=float,default=0.30)
    ap.add_argument("--arm-amp",type=float,default=0.25)
    ap.add_argument("--ramp-time",type=float,default=1.0)
    ap.add_argument("--start-time",type=float,default=0.5)
    ap.add_argument("--ik-iters",type=int,default=30)
    a=ap.parse_args()
    p=dict(vx=a.vx,height=a.height_com,step_time=a.step_time,clearance=a.clearance,kx=a.kx,ky=a.ky,
           max_step=a.max_step,arm_amp=a.arm_amp,ramp_time=a.ramp_time,start_time=a.start_time,ik_iters=a.ik_iters)
    simulate(a.model,p,a.duration,a.video,a.metrics,a.width,a.height,a.fps)

if __name__=="__main__": main()
