// rnn.js: the built-in RNN baseline opponent for the Slime Volley court.
//
// slimevolleygym's reference net (~120 params): the fixed opponent the policies play against. Kept in JS so
// it runs instantly without Pyodide. obs = 12-d normalized vector [me xyz.., ball.., opp..] / 10, from the
// acting agent's own perspective (mirrored for the left side); act() returns [forward, backward, jump].
// getActs() exposes the input + hidden activations, which the demo lights up as it plays.

// ----- the built-in RNN baseline (120 params): slimevolleygym's BaselinePolicy, the fixed opponent.
const RNN_W = [
  7.5719,4.4285,2.2716,-0.3598,-7.8189,-2.5422,-3.2034,0.3935,1.2202,-0.49,-0.0316,0.5221,0.7026,0.4179,-2.1689,
  1.646,-13.3639,1.5151,1.1175,-5.3561,5.0442,0.8451,0.3987,-2.9501,-3.7811,-5.8994,6.4167,2.5014,7.338,-2.9887,
  2.4586,13.4191,2.7395,-3.9708,1.6548,-2.7554,-1.5345,-6.4708,9.2426,-0.7392,0.4452,1.8828,-2.6277,-10.851,-3.2353,
  -4.4653,-3.1153,-1.3707,7.318,16.0902,1.4686,7.0391,1.7765,-1.155,2.6697,-8.8877,1.1958,-3.2839,-5.4425,1.6809,
  7.6812,-2.4732,1.738,0.3781,0.8718,2.5886,1.6911,1.2953,-9.0052,-4.6038,-6.7447,-2.5528,0.4391,-4.9278,-3.6695,
  -4.8673,-1.6035,1.5011,-5.6124,4.9747,1.8998,3.0359,6.2983,-4.8568,-2.1888,-4.1143,-3.9874,-0.0459,4.7134,2.8952,
  -9.3627,-4.685,0.3601,-1.3699,9.7294,11.5596,0.1918,3.0783,0.0329,-0.1362,-0.1188,-0.7579,0.3278,-0.977,-0.9377,
];
const RNN_B = [2.2935,-2.0353,-1.7786,5.4567,-3.6368,3.4996,-0.0685];

function makeRNN(thresh) {
  thresh = thresh || [0.75, 0.75, 0.75];
  let out = new Array(7).fill(0);            // recurrent state carried between frames
  const blank = () => ({ obs: new Array(8).fill(0), state: new Array(7).fill(0), logits: new Array(7).fill(0), fired: [false, false, false], thresh });
  let acts = blank();
  return {
    rnn: true,                               // viz hint: render activations, not a code path
    getActs: () => acts,
    reset() { out = new Array(7).fill(0); acts = blank(); },
    act(obs) {
      const state = out.slice();             // the recurrent state fed IN this frame (= last frame's logits)
      const inp = [obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], ...out];
      const no = new Array(7);
      for (let i = 0; i < 7; i++) {
        let s = RNN_B[i];
        for (let j = 0; j < 15; j++) s += RNN_W[i * 15 + j] * inp[j];
        no[i] = Math.tanh(s);
      }
      out = no;
      const fired = [out[0] > thresh[0], out[1] > thresh[1], out[2] > thresh[2]];
      acts = { obs: obs.slice(0, 8), state, logits: no.slice(), fired, thresh };
      return [fired[0] ? 1 : 0, fired[1] ? 1 : 0, fired[2] ? 1 : 0];
    },
  };
}

// The local (JS) actor registry, just the RNN baseline. (The learned policies run their real Python via
// Pyodide instead; see pyrunner.js.)
export const POLICY_FACTORY = {
  rnn_baseline: makeRNN,                              // the built-in baseline opponent
};
