import argparse, math, json
from pathlib import Path
import numpy as np
import mujoco


def quat_to_rpy(q):
    # MuJoCo quaternion w,x,y,z to roll/pitch/yaw in radians.
    w, x, y, z = q
    t0 = 2.0*(w*x + y*z)
    t1 = 1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(t0, t1)
    t2 = 2.0*(w*y - z*x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x*x*(3 - 2*x)

class GaitController:
    def __init__(self, model, data, params):
        self.model = model
        self.params = params
        ki = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'stand')
        self.stand_ctrl = model.key_ctrl[ki].copy()
        self.prev = self.stand_ctrl.copy()
        self.act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)}
        self.jq = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j] for j in range(model.njnt)}
        self.jv = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_dofadr[j] for j in range(model.njnt)}
        self.ctrl_min = model.actuator_ctrlrange[:,0].copy()
        self.ctrl_max = model.actuator_ctrlrange[:,1].copy()

    def _set(self, ctrl, name, value):
        if name in self.act:
            i = self.act[name]
            ctrl[i] = min(self.ctrl_max[i], max(self.ctrl_min[i], value))

    def _leg(self, ctrl, prefix, side, phase, amp, lateral, knee_swing, knee_stance, ankle_gain, ramp):
        p = self.params
        duty = p['duty']
        if phase < duty:
            u = phase / duty
            h = -amp + 2.0*amp*smoothstep(u)  # foot moves from front to back relative pelvis.
            knee = knee_stance + 0.035*math.sin(math.pi*u)**2
            toe = 0.03*math.sin(math.pi*u)
            support = 1.0
        else:
            u = (phase - duty) / (1.0 - duty)
            h = amp - 2.0*amp*smoothstep(u)  # swing back to front.
            # higher knee in mid-swing for toe clearance
            knee = knee_stance + knee_swing*(math.sin(math.pi*u)**1.15)
            toe = -0.08*math.sin(math.pi*u)
            support = 0.0
        # Lateral pelvis shift over the stance leg. side=+1 left, -1 right.
        # Stance foot relative-y is reduced toward zero, which moves the pelvis over that foot.
        roll = -side * lateral * (0.65 + 0.35*support)
        if support < 0.5:
            # swing leg abducts slightly for clearance but not too much.
            roll = side * 0.02
        ankle = p.get('ankle_bias', 0.0) - ankle_gain*h - 0.26*knee + toe
        hip = p.get('hip_bias', 0.0) + h
        self._set(ctrl, f'{prefix}_hip_pitch_joint', ramp*hip)
        self._set(ctrl, f'{prefix}_knee_joint', ramp*knee)
        self._set(ctrl, f'{prefix}_ankle_pitch_joint', ramp*ankle)
        self._set(ctrl, f'{prefix}_hip_roll_joint', ramp*roll)
        self._set(ctrl, f'{prefix}_ankle_roll_joint', ramp*(0.55*roll))
        self._set(ctrl, f'{prefix}_hip_yaw_joint', ramp*(side*0.018*math.sin(2*math.pi*phase)))

    def compute(self, data):
        p = self.params
        t = data.time
        ctrl = self.stand_ctrl.copy()
        ramp = smoothstep(min(1.0, max(0.0, (t-p['start_time'])/p['ramp_time'])))
        # desired CPG phase, start with left stance and right swing.
        cyc = p['cycle']
        phi = ((t - p['start_time']) / cyc) % 1.0
        amp = p['hip_amp']
        lateral = p['lateral']
        self._leg(ctrl, 'left', +1, phi, amp, lateral, p['knee_swing'], p['knee_stance'], p['ankle_gain'], ramp)
        self._leg(ctrl, 'right', -1, (phi + 0.5) % 1.0, amp, lateral, p['knee_swing'], p['knee_stance'], p['ankle_gain'], ramp)
        # Arm counter-swing using existing shoulder position actuators, no new properties.
        arm = ramp * p['arm_amp'] * math.sin(2*math.pi*phi)
        self._set(ctrl, 'left_shoulder_pitch_joint', 0.2 + arm)
        self._set(ctrl, 'right_shoulder_pitch_joint', 0.2 - arm)
        self._set(ctrl, 'left_shoulder_roll_joint', 0.2)
        self._set(ctrl, 'right_shoulder_roll_joint', -0.2)
        self._set(ctrl, 'left_elbow_joint', 1.28 - 0.18*abs(arm))
        self._set(ctrl, 'right_elbow_joint', 1.28 - 0.18*abs(arm))
        # Torso feedback: keep pelvis upright by shaping hip/ankle targets, not by external forces.
        roll, pitch, yaw = quat_to_rpy(data.qpos[3:7])
        # Velocity target regulation: lean/step a little more when too slow, less when too fast.
        vx = data.qvel[0]
        speed_err = p['target_speed'] - vx
        # limit feedback to avoid stiff/nonphysical lurches
        pitch_fb = max(-0.10, min(0.10, p['pitch_k']*pitch + p['pitch_d']*data.qvel[4] - p['speed_k']*speed_err))
        roll_fb = max(-0.08, min(0.08, p['roll_k']*roll + p['roll_d']*data.qvel[3]))
        for pref in ['left','right']:
            self._set(ctrl, f'{pref}_hip_pitch_joint', ctrl[self.act[f'{pref}_hip_pitch_joint']] + pitch_fb)
            self._set(ctrl, f'{pref}_ankle_pitch_joint', ctrl[self.act[f'{pref}_ankle_pitch_joint']] - 0.45*pitch_fb)
            self._set(ctrl, f'{pref}_hip_roll_joint', ctrl[self.act[f'{pref}_hip_roll_joint']] + roll_fb)
            self._set(ctrl, f'{pref}_ankle_roll_joint', ctrl[self.act[f'{pref}_ankle_roll_joint']] - 0.35*roll_fb)
        self._set(ctrl, 'waist_pitch_joint', max(-0.12, min(0.12, -0.35*pitch)))
        self._set(ctrl, 'waist_roll_joint', max(-0.12, min(0.12, -0.35*roll)))
        # Low-pass desired motor positions to leave compliance at contacts.
        tau = p['filter_tau']
        alpha = 1.0 if tau <= 0 else min(1.0, self.model.opt.timestep / tau)
        self.prev += alpha * (ctrl - self.prev)
        return self.prev

def key_id(model):
    for name in ('stand','home','knees_bent'):
        kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            return kid
    return 0

def run(params, duration=10.0, model_path='third_party/mujoco_menagerie/unitree_g1/scene.xml'):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    ki = key_id(model)
    mujoco.mj_resetDataKeyframe(model, data, ki)
    ctrl = GaitController(model,data,params)
    startx = float(data.qpos[0])
    max_tilt=0; minz=9; maxy=0; fallen=False
    samples=[]
    steps=int(duration/model.opt.timestep)
    for step in range(steps):
        data.ctrl[:] = ctrl.compute(data)
        mujoco.mj_step(model, data)
        roll,pitch,yaw=quat_to_rpy(data.qpos[3:7])
        tilt=max(abs(roll),abs(pitch)); max_tilt=max(max_tilt,tilt); minz=min(minz,float(data.qpos[2])); maxy=max(maxy,abs(float(data.qpos[1])))
        if data.qpos[2] < params.get('fall_z',0.45) or tilt > params.get('fall_tilt',1.0):
            fallen=True
            # don't break; continue to see artifact, but mark
        if step % int(0.5/model.opt.timestep)==0:
            samples.append((float(data.time),float(data.qpos[0]),float(data.qpos[1]),float(data.qpos[2]),float(data.qvel[0]),roll,pitch,data.ncon))
    dist=float(data.qpos[0]-startx)
    return {'duration':float(data.time),'distance':dist,'avg_speed':dist/data.time,'final_x':float(data.qpos[0]),'final_y':float(data.qpos[1]),'final_z':float(data.qpos[2]),'min_z':minz,'max_abs_y':maxy,'max_tilt_rad':max_tilt,'fallen':fallen,'samples':samples}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--json',type=str)
    ap.add_argument('--duration',type=float,default=10)
    args=ap.parse_args()
    params={
        'target_speed':0.5,'cycle':0.72,'duty':0.62,'hip_amp':0.24,'hip_bias':0.0,'knee_swing':0.46,'knee_stance':0.035,
        'ankle_gain':0.7,'ankle_bias':0.0,'lateral':0.12,'arm_amp':0.35,'start_time':0.5,'ramp_time':1.0,'filter_tau':0.025,
        'pitch_k':0.25,'pitch_d':0.03,'speed_k':0.08,'roll_k':0.35,'roll_d':0.025,
    }
    if args.json:
        params.update(json.loads(args.json))
    print(json.dumps({'params':params,'metrics':run(params,args.duration)},indent=2))
if __name__=='__main__': main()
