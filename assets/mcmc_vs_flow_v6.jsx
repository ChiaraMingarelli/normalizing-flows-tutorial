import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ============================================================
// Seeded PRNG
// ============================================================
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    var t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function seededGaussian(rng) {
  const u1 = rng(), u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

// ============================================================
// Distribution definitions
// ============================================================
const SIGMA_OBS = 0.8, N_DATA = 8;
const rngData = mulberry32(77);
const tObs = Array.from({ length: N_DATA }, (_, i) => (i + 0.5) / N_DATA);
const dataObs = tObs.map(t => 2.2 * Math.sin(2 * Math.PI * t + 1.8) + SIGMA_OBS * seededGaussian(rngData));

function makeDistributions() {
  return {
    sinusoid: {
      name: "Sinusoid",
      desc: "d(t) = A sin(2πt+φ) + noise",
      detail: "Curved ridge from amplitude–phase degeneracy",
      labels: ["A", "φ"],
      ranges: [[0.2, 4.5], [0, 2 * Math.PI]],
      trueVals: [2.2, 1.8],
      mcmcStart: [1.0, 4.0],
      propScale: 0.25,
      logP(p0, p1) {
        if (p0 < 0.2 || p0 > 4.5 || p1 < 0 || p1 > 2 * Math.PI) return -Infinity;
        let ll = 0;
        for (let i = 0; i < N_DATA; i++) { const r = dataObs[i] - p0 * Math.sin(2 * Math.PI * tObs[i] + p1); ll -= r * r / (2 * SIGMA_OBS * SIGMA_OBS); }
        return ll;
      }
    },
    bimodal: {
      name: "Bimodal",
      desc: "Two separated Gaussian modes",
      detail: "MCMC struggles to jump between modes",
      labels: ["x₁", "x₂"],
      ranges: [[-6, 6], [-6, 6]],
      trueVals: [null, null],
      mcmcStart: [-3, 2],
      propScale: 0.5,
      logP(x, y) {
        if (x < -6 || x > 6 || y < -6 || y > 6) return -Infinity;
        const g1 = Math.exp(-((x - 2.5) ** 2 + (y - 2.5) ** 2) / (2 * 0.7 * 0.7));
        const g2 = Math.exp(-((x + 2.5) ** 2 + (y + 2.5) ** 2) / (2 * 0.7 * 0.7));
        return Math.log(g1 + g2 + 1e-30);
      }
    },
    banana: {
      name: "Banana",
      desc: "Rosenbrock-like posterior",
      detail: "Strong nonlinear correlation; hard to propose along the ridge",
      labels: ["x₁", "x₂"],
      ranges: [[-2.5, 4.5], [-2, 12]],
      trueVals: [1.0, 1.0],
      mcmcStart: [0.0, 4.0],
      propScale: 0.3,
      logP(x, y) {
        if (x < -2.5 || x > 4.5 || y < -2 || y > 12) return -Infinity;
        return -((x - 1) ** 2) / 2 - 4 * ((y - x * x) ** 2);
      }
    },
    triplemode: {
      name: "Triple mode",
      desc: "Three overlapping correlated Gaussians",
      detail: "Asymmetric modes with different orientations; tests mode-hopping and coverage",
      labels: ["x₁", "x₂"],
      ranges: [[-4, 6], [-4, 6]],
      trueVals: [null, null],
      mcmcStart: [0.0, 0.0],
      propScale: 0.5,
      logP(x, y) {
        if (x < -4 || x > 6 || y < -4 || y > 6) return -Infinity;
        // Mode 1: centered (0, 0), elongated along x=y
        const dx1 = x, dy1 = y;
        const g1 = Math.exp(-0.5 * (1.2 * dx1 * dx1 - 0.8 * dx1 * dy1 + 1.2 * dy1 * dy1));
        // Mode 2: centered (2.5, 2.5), elongated along x=-y
        const dx2 = x - 2.5, dy2 = y - 2.5;
        const g2 = 0.8 * Math.exp(-0.5 * (1.5 * dx2 * dx2 + 1.0 * dx2 * dy2 + 1.5 * dy2 * dy2));
        // Mode 3: centered (1, -1.5), rounder, wider
        const dx3 = x - 1, dy3 = y + 1.5;
        const g3 = 0.6 * Math.exp(-0.5 * (0.6 * dx3 * dx3 + 0.6 * dy3 * dy3));
        return Math.log(g1 + g2 + g3 + 1e-30);
      }
    },
    funnel: {
      name: "Funnel",
      desc: "Neal's funnel",
      detail: "Hierarchical pathology: narrow neck + wide base. Notoriously hard for MCMC.",
      labels: ["x₁", "x₂"],
      ranges: [[-8, 8], [-4, 4]],
      trueVals: [null, null],
      mcmcStart: [0.0, 0.0],
      propScale: 0.4,
      logP(x, v) {
        if (x < -8 || x > 8 || v < -4 || v > 4) return -Infinity;
        // v ~ N(0, 3), x | v ~ N(0, exp(v))
        const logpv = -(v * v) / (2 * 3 * 3);
        const sigma2 = Math.exp(v);
        const logpx = -0.5 * Math.log(2 * Math.PI * sigma2) - (x * x) / (2 * sigma2);
        return logpv + logpx;
      }
    },
  };
}
const DISTRIBUTIONS = makeDistributions();

// ============================================================
// Exact posterior grid + marginals (generic)
// ============================================================
const GRID_N = 80;
function computeGrid(dist) {
  const grid = new Float64Array(GRID_N * GRID_N);
  const [r0, r1] = dist.ranges;
  const d0 = (r0[1] - r0[0]) / GRID_N, d1 = (r1[1] - r1[0]) / GRID_N;
  let mx = -Infinity;
  for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) {
    const lp = dist.logP(r0[0] + (i + .5) * d0, r1[0] + (j + .5) * d1);
    grid[j * GRID_N + i] = lp; if (lp > mx) mx = lp;
  }
  const prob = new Float64Array(GRID_N * GRID_N);
  for (let i = 0; i < grid.length; i++) prob[i] = Math.exp(grid[i] - mx);
  return prob;
}
function computeExactMarginals(grid) {
  const m0 = new Float64Array(GRID_N), m1 = new Float64Array(GRID_N);
  for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) { m0[i] += grid[j * GRID_N + i]; m1[j] += grid[j * GRID_N + i]; }
  let s0 = 0, s1 = 0;
  for (let i = 0; i < GRID_N; i++) { s0 += m0[i]; s1 += m1[i]; }
  for (let i = 0; i < GRID_N; i++) { m0[i] /= s0; m1[i] /= s1; }
  return { marg0: m0, marg1: m1 };
}

// ============================================================
// MCMC (generic)
// ============================================================
const TOTAL_MCMC = 6000;
function buildChain(dist, seed = 123) {
  const rng = mulberry32(seed);
  let [p0, p1] = dist.mcmcStart;
  let cLP = dist.logP(p0, p1);
  const chain = [{ p0, p1 }];
  const ps = dist.propScale;
  for (let s = 0; s < TOTAL_MCMC; s++) {
    const pp0 = p0 + ps * seededGaussian(rng), pp1 = p1 + ps * seededGaussian(rng);
    const pLP = dist.logP(pp0, pp1);
    if (Math.log(rng()) < pLP - cLP) { p0 = pp0; p1 = pp1; cLP = pLP; }
    chain.push({ p0, p1 });
  }
  return chain;
}

// ============================================================
// Flow (generic — uses dist.logP)
// ============================================================
function initFlowParams(nL, seed = 55) {
  const rng = mulberry32(seed);
  return Array.from({ length: nL }, () => ({ u: [.3 * seededGaussian(rng), .3 * seededGaussian(rng)], w: [.3 * seededGaussian(rng), .3 * seededGaussian(rng)], b: .1 * seededGaussian(rng) }));
}
function flowForward(z, params) {
  let x = [...z], ld = 0;
  for (const { u, w, b } of params) {
    const dot = w[0] * x[0] + w[1] * x[1] + b, th = Math.tanh(dot), dth = 1 - th * th;
    ld += Math.log(Math.abs(1 + u[0] * dth * w[0] + u[1] * dth * w[1]) + 1e-10);
    x = [x[0] + u[0] * th, x[1] + u[1] * th];
  }
  return { x, logDet: ld };
}
function flowToPhysical(x, ranges) {
  return [
    ranges[0][0] + (ranges[0][1] - ranges[0][0]) / (1 + Math.exp(-x[0])),
    ranges[1][0] + (ranges[1][1] - ranges[1][0]) / (1 + Math.exp(-x[1])),
  ];
}
function logSJ(x, rng) {
  const s = 1 / (1 + Math.exp(-x));
  return Math.log(s * (1 - s) + 1e-10) + Math.log(rng[1] - rng[0]);
}

function trainStep(params, bs, lr, rng, dist) {
  const eps = 1e-4;
  const zB = Array.from({ length: bs }, () => [seededGaussian(rng), seededGaussian(rng)]);
  function loss(p) {
    let t = 0;
    for (const z of zB) {
      const { x, logDet } = flowForward(z, p);
      const phys = flowToPhysical(x, dist.ranges);
      t += -logDet - logSJ(x[0], dist.ranges[0]) - logSJ(x[1], dist.ranges[1]) - dist.logP(phys[0], phys[1]);
    }
    return t / bs;
  }
  const bL = loss(params);
  const nP = params.map(l => ({ u: [...l.u], w: [...l.w], b: l.b }));
  for (let l = 0; l < params.length; l++) {
    for (let d = 0; d < 2; d++) {
      nP[l].u[d] += eps; const a = loss(nP); nP[l].u[d] -= 2 * eps; const b = loss(nP); nP[l].u[d] += eps; nP[l].u[d] -= lr * Math.max(-2, Math.min(2, (a - b) / (2 * eps)));
      nP[l].w[d] += eps; const c = loss(nP); nP[l].w[d] -= 2 * eps; const e = loss(nP); nP[l].w[d] += eps; nP[l].w[d] -= lr * Math.max(-2, Math.min(2, (c - e) / (2 * eps)));
    }
    nP[l].b += eps; const a = loss(nP); nP[l].b -= 2 * eps; const b2 = loss(nP); nP[l].b += eps; nP[l].b -= lr * Math.max(-2, Math.min(2, (a - b2) / (2 * eps)));
  }
  return { params: nP, loss: bL };
}
function sampleFlowDist(params, n, seed, dist) {
  const rng = mulberry32(seed), out = [];
  const [r0, r1] = dist.ranges;
  for (let i = 0; i < n; i++) {
    const z = [seededGaussian(rng), seededGaussian(rng)];
    const { x } = flowForward(z, params);
    const phys = flowToPhysical(x, dist.ranges);
    if (phys[0] >= r0[0] && phys[0] <= r0[1] && phys[1] >= r1[0] && phys[1] <= r1[1]) out.push(phys);
  }
  return out;
}

const TOTAL_EPOCHS = 1200, SNAP_INT = 8;
function precomputeSnapsAsync(nL, dist, onProgress, onDone) {
  let p = initFlowParams(nL, 55); const rng = mulberry32(999);
  const snaps = [{ epoch: 0, params: JSON.parse(JSON.stringify(p)), loss: null }];
  const lossHist = [];
  const bs = nL > 10 ? 60 : 100;
  let e = 1; const CHUNK = nL > 10 ? 10 : 20;
  function step() {
    const end = Math.min(e + CHUNK, TOTAL_EPOCHS + 1);
    for (; e < end; e++) {
      const r = trainStep(p, bs, .02 * Math.exp(-e * .002), rng, dist); p = r.params;
      lossHist.push({ epoch: e, loss: r.loss });
      if (e % SNAP_INT === 0 || e === TOTAL_EPOCHS) snaps.push({ epoch: e, params: JSON.parse(JSON.stringify(p)), loss: r.loss });
    }
    if (e <= TOTAL_EPOCHS) { if (onProgress) onProgress(e, TOTAL_EPOCHS); requestAnimationFrame(step); }
    else onDone(snaps, lossHist);
  }
  requestAnimationFrame(step);
}

// ============================================================
// ESS
// ============================================================
function computeESS(chain, burnin) {
  if (chain.length <= burnin + 10) return 0;
  const vals = chain.slice(burnin).map(s => s.p0);
  const n = vals.length;
  let mean = 0; for (let i = 0; i < n; i++) mean += vals[i]; mean /= n;
  let c0 = 0; for (let i = 0; i < n; i++) c0 += (vals[i] - mean) ** 2; c0 /= n;
  if (c0 === 0) return n;
  let tauSum = 0;
  for (let lag = 1; lag < Math.min(n, 500); lag++) {
    let ck = 0; for (let i = 0; i < n - lag; i++) ck += (vals[i] - mean) * (vals[i + lag] - mean); ck /= n;
    if (ck / c0 < 0.05) break; tauSum += ck / c0;
  }
  return Math.max(1, Math.round(n / (1 + 2 * tauSum)));
}

// ============================================================
// Viridis
// ============================================================
const VIR = [[68,1,84],[72,34,115],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37]];
function viridis(t) { t = Math.max(0, Math.min(1, t)); const idx = t * (VIR.length - 1), lo = Math.floor(idx), hi = Math.min(lo + 1, VIR.length - 1), f = idx - lo; return VIR[lo].map((c, i) => Math.round(c + f * (VIR[hi][i] - c))); }

// ============================================================
// Layout
// ============================================================
const SIDE_W = 150, COL_W = 310, P2D_H = 250, MH = 105, LOSS_H = 95;
const PAD = { top: 26, right: 14, bottom: 30, left: 40 };
function plotArea(w, h) { return { x: PAD.left, y: PAD.top, w: w - PAD.left - PAD.right, h: h - PAD.top - PAD.bottom }; }

// ============================================================
// Component
// ============================================================
export default function MCMCvsFlow() {
  const [distKey, setDistKey] = useState("sinusoid");
  const dist = DISTRIBUTIONS[distKey];

  const mcmc2dRef = useRef(null), flow2dRef = useRef(null);
  const mcmcM0Ref = useRef(null), mcmcM1Ref = useRef(null);
  const flowM0Ref = useRef(null), flowM1Ref = useRef(null);
  const lossRef = useRef(null);
  const bgRef = useRef(null);

  const [mcmcFrame, setMcmcFrame] = useState(0);
  const [mcmcPlaying, setMcmcPlaying] = useState(false);
  const [mcmcSpeed, setMcmcSpeed] = useState(2);

  const [flowPhase, setFlowPhase] = useState("idle");
  const [flowEpochDisplay, setFlowEpochDisplay] = useState(0);
  const [nLayers, setNLayers] = useState(8);
  const [computing, setComputing] = useState(false);
  const [trainProgress, setTrainProgress] = useState(null);

  const mcmcAnimRef = useRef(null), flowAnimRef = useRef(null), layerGenRef = useRef(0);
  const MCMC_FRAMES = 1000;

  // Recompute grid + chain when dist changes
  const pGrid = useMemo(() => computeGrid(dist), [distKey]);
  const { marg0: ex0, marg1: ex1 } = useMemo(() => computeExactMarginals(pGrid), [pGrid]);
  const fullChain = useMemo(() => buildChain(dist, 123), [distKey]);

  const [flowSnaps, setFlowSnaps] = useState(null);
  const [lossHistory, setLossHistory] = useState([]);
  const [flowSamples, setFlowSamples] = useState([]);
  const [flowTrainedParams, setFlowTrainedParams] = useState(null);

  // Reset flow when dist or layers change
  useEffect(() => {
    setComputing(true);
    setFlowPhase("idle"); setFlowSamples([]); setFlowEpochDisplay(0);
    setTrainProgress(null); setLossHistory([]); setFlowTrainedParams(null);
    const gen = ++layerGenRef.current;
    precomputeSnapsAsync(nLayers, dist,
      (e, total) => { if (gen === layerGenRef.current) setTrainProgress(Math.round(e / total * 100)); },
      (snaps, lh) => { if (gen === layerGenRef.current) { setFlowSnaps(snaps); setLossHistory(lh); setComputing(false); setTrainProgress(null); } }
    );
    // Also reset MCMC view
    setMcmcFrame(0); setMcmcPlaying(false);
  }, [distKey, nLayers]);

  // BG image
  useEffect(() => {
    const img = new ImageData(GRID_N, GRID_N); let mx = 0;
    for (let i = 0; i < pGrid.length; i++) if (pGrid[i] > mx) mx = pGrid[i];
    for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) {
      const v = mx > 0 ? Math.pow(pGrid[j * GRID_N + i] / mx, .6) : 0; const [r, g, b] = viridis(v); const idx = (j * GRID_N + i) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
    createImageBitmap(img).then(bmp => { const c = document.createElement("canvas"); c.width = GRID_N; c.height = GRID_N; c.getContext("2d").drawImage(bmp, 0, 0); bgRef.current = c; setMcmcFrame(f => f); /* force redraw */ });
  }, [pGrid]);

  // MCMC animation
  useEffect(() => {
    if (!mcmcPlaying) { if (mcmcAnimRef.current) cancelAnimationFrame(mcmcAnimRef.current); return; }
    const tick = () => { setMcmcFrame(p => { const n = p + mcmcSpeed; if (n >= MCMC_FRAMES) { setMcmcPlaying(false); return MCMC_FRAMES; } return n; }); mcmcAnimRef.current = requestAnimationFrame(tick); };
    mcmcAnimRef.current = requestAnimationFrame(tick);
    return () => { if (mcmcAnimRef.current) cancelAnimationFrame(mcmcAnimRef.current); };
  }, [mcmcPlaying, mcmcSpeed]);

  // Flow training animation
  useEffect(() => {
    if (flowPhase !== "training" || !flowSnaps) { if (flowAnimRef.current) cancelAnimationFrame(flowAnimRef.current); return; }
    let idx = 0;
    const tick = () => {
      idx += 2;
      if (idx >= flowSnaps.length) {
        const fin = flowSnaps[flowSnaps.length - 1];
        setFlowEpochDisplay(fin.epoch); setFlowTrainedParams(fin.params); setFlowPhase("trained"); return;
      }
      setFlowEpochDisplay(flowSnaps[idx].epoch);
      setFlowSamples(sampleFlowDist(flowSnaps[idx].params, 800, 42, dist));
      flowAnimRef.current = requestAnimationFrame(tick);
    };
    flowAnimRef.current = requestAnimationFrame(tick);
    return () => { if (flowAnimRef.current) cancelAnimationFrame(flowAnimRef.current); };
  }, [flowPhase, flowSnaps]);

  // Flow sampling animation
  useEffect(() => {
    if (flowPhase !== "sampling" || !flowTrainedParams) return;
    let count = 0, acc = [];
    const tick = () => {
      const rng = mulberry32(42 + count);
      for (let i = 0; i < 100; i++) {
        const z = [seededGaussian(rng), seededGaussian(rng)];
        const { x } = flowForward(z, flowTrainedParams);
        const phys = flowToPhysical(x, dist.ranges);
        if (phys[0] >= dist.ranges[0][0] && phys[0] <= dist.ranges[0][1] && phys[1] >= dist.ranges[1][0] && phys[1] <= dist.ranges[1][1]) acc.push(phys);
      }
      count += 100; setFlowSamples([...acc]);
      if (acc.length < 1500) flowAnimRef.current = requestAnimationFrame(tick);
    };
    flowAnimRef.current = requestAnimationFrame(tick);
    return () => { if (flowAnimRef.current) cancelAnimationFrame(flowAnimRef.current); };
  }, [flowPhase, flowTrainedParams]);

  const mcmcStep = Math.min(Math.round((mcmcFrame / MCMC_FRAMES) * TOTAL_MCMC), TOTAL_MCMC);

  const toPlot = useCallback((v0, v1, p, ranges) => [
    p.x + ((v0 - ranges[0][0]) / (ranges[0][1] - ranges[0][0])) * p.w,
    p.y + ((v1 - ranges[1][0]) / (ranges[1][1] - ranges[1][0])) * p.h
  ], []);

  function draw2dBg(ctx, p, w, h, title, tColor) {
    ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, w, h);
    if (bgRef.current) { ctx.globalAlpha = .3; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(bgRef.current, p.x, p.y, p.w, p.h); ctx.globalAlpha = 1; }
    ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, p.y + p.h); ctx.lineTo(p.x + p.w, p.y + p.h); ctx.stroke();
    ctx.fillStyle = "#506878"; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText(dist.labels[0], p.x + p.w / 2, h - 4);
    ctx.save(); ctx.translate(11, p.y + p.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(dist.labels[1], 0, 0); ctx.restore();
    ctx.fillStyle = tColor; ctx.font = "11px 'IBM Plex Mono',monospace"; ctx.textAlign = "left"; ctx.fillText(title, p.x + 3, p.y - 9);
    // Ticks
    ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.fillStyle = "#354555"; ctx.textAlign = "center";
    const [r0, r1] = dist.ranges;
    const step0 = niceStep(r0[0], r0[1]), step1 = niceStep(r1[0], r1[1]);
    for (let v = Math.ceil(r0[0] / step0) * step0; v <= r0[1]; v += step0) { const [x] = toPlot(v, 0, p, dist.ranges); ctx.fillText(fmtTick(v), x, p.y + p.h + 11); }
    ctx.textAlign = "right";
    for (let v = Math.ceil(r1[0] / step1) * step1; v <= r1[1]; v += step1) { const [, y] = toPlot(0, v, p, dist.ranges); ctx.fillText(fmtTick(v), p.x - 4, y + 3); }
    // True crosshair
    if (dist.trueVals[0] != null) {
      const [tx, ty] = toPlot(dist.trueVals[0], dist.trueVals[1], p, dist.ranges);
      ctx.strokeStyle = "rgba(255,255,255,.12)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(tx, p.y); ctx.lineTo(tx, p.y + p.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p.x, ty); ctx.lineTo(p.x + p.w, ty); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  // MCMC 2D
  useEffect(() => {
    const c = mcmc2dRef.current; if (!c) return; const ctx = c.getContext("2d"); const p = plotArea(COL_W, P2D_H);
    draw2dBg(ctx, p, COL_W, P2D_H, "MCMC (Metropolis-Hastings)", "#e8a060");
    const n = mcmcStep, ts = Math.max(0, n - 500);
    ctx.strokeStyle = "rgba(255,255,255,.1)"; ctx.lineWidth = .7; ctx.beginPath();
    for (let i = ts; i <= n; i++) { const [x, y] = toPlot(fullChain[i].p0, fullChain[i].p1, p, dist.ranges); i === ts ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
    if (n > 200) { ctx.fillStyle = "rgba(255,130,70,.25)"; const th = Math.max(1, Math.floor((n - 200) / 1500)); for (let i = 200; i <= n; i += th) { const [x, y] = toPlot(fullChain[i].p0, fullChain[i].p1, p, dist.ranges); ctx.beginPath(); ctx.arc(x, y, 1.3, 0, 2 * Math.PI); ctx.fill(); } }
    if (n > 0) { const [cx, cy] = toPlot(fullChain[n].p0, fullChain[n].p1, p, dist.ranges); const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 10); g.addColorStop(0, "rgba(255,180,80,.6)"); g.addColorStop(1, "rgba(255,180,80,0)"); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, 10, 0, 2 * Math.PI); ctx.fill(); ctx.fillStyle = "#ffb050"; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2 * Math.PI); ctx.fill(); }
    ctx.fillStyle = "#605848"; ctx.font = "9px 'IBM Plex Mono',monospace"; ctx.textAlign = "right";
    ctx.fillText(`step ${n}`, p.x + p.w - 2, p.y + 11);
    const ess = computeESS(fullChain.slice(0, n + 1), 200);
    if (n > 200) { ctx.fillText(`samples: ${n - 200}`, p.x + p.w - 2, p.y + 22); ctx.fillStyle = "#e8a060"; ctx.fillText(`ESS: ${ess}`, p.x + p.w - 2, p.y + 33); }
  }, [mcmcFrame, distKey, fullChain]);

  // Flow 2D
  useEffect(() => {
    const c = flow2dRef.current; if (!c) return; const ctx = c.getContext("2d"); const p = plotArea(COL_W, P2D_H);
    draw2dBg(ctx, p, COL_W, P2D_H, `Normalizing Flow (${nLayers}L · ${nLayers * 5}p)`, "#70c8a0");
    if (computing) { ctx.fillStyle = "rgba(12,12,32,.7)"; ctx.fillRect(p.x, p.y, p.w, p.h); ctx.fillStyle = "#70c8a0"; ctx.font = "12px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText(`Precomputing... ${trainProgress ?? 0}%`, p.x + p.w / 2, p.y + p.h / 2); return; }
    if (flowPhase === "idle") { ctx.fillStyle = "rgba(12,12,32,.6)"; ctx.fillRect(p.x, p.y, p.w, p.h); ctx.fillStyle = "#506878"; ctx.font = "12px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText("Press Train ▸", p.x + p.w / 2, p.y + p.h / 2); return; }
    ctx.fillStyle = flowPhase === "training" ? "rgba(100,210,130,.18)" : "rgba(100,210,130,.3)";
    for (const s of flowSamples) { const [x, y] = toPlot(s[0], s[1], p, dist.ranges); ctx.beginPath(); ctx.arc(x, y, 1.4, 0, 2 * Math.PI); ctx.fill(); }
    ctx.font = "9px 'IBM Plex Mono',monospace"; ctx.textAlign = "right";
    if (flowPhase === "training") { ctx.fillStyle = "#c0a050"; ctx.fillText(`TRAINING epoch ${flowEpochDisplay}`, p.x + p.w - 2, p.y + 11); }
    else if (flowPhase === "trained") { ctx.fillStyle = "#70c8a0"; ctx.fillText("TRAINED — press Sample ▸", p.x + p.w - 2, p.y + 11); }
    else if (flowPhase === "sampling") { ctx.fillStyle = "#506858"; ctx.fillText(`samples: ${flowSamples.length}`, p.x + p.w - 2, p.y + 11); ctx.fillStyle = "#70c8a0"; ctx.fillText(`ESS: ${flowSamples.length}`, p.x + p.w - 2, p.y + 22); }
  }, [flowPhase, flowSamples, flowEpochDisplay, computing, trainProgress, nLayers, distKey]);

  // Loss curve
  useEffect(() => {
    const c = lossRef.current; if (!c) return; const ctx = c.getContext("2d");
    const cW = c.width, cH = c.height;
    const m = { top: 12, right: 14, bottom: 20, left: 50 }; const pW = cW - m.left - m.right, pH = cH - m.top - m.bottom;
    ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, cW, cH);
    ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
    ctx.fillStyle = "#70c8a0"; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textAlign = "left"; ctx.fillText("Training loss (KL divergence)", m.left + 3, m.top + 9);
    if (lossHistory.length === 0) { ctx.fillStyle = "#354555"; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText(computing ? "Computing..." : "", m.left + pW / 2, m.top + pH / 2); return; }
    const losses = lossHistory.map(l => l.loss).filter(l => l != null && isFinite(l)); if (losses.length < 2) return;
    let minL = Infinity, maxL = -Infinity; for (const l of losses) { if (l < minL) minL = l; if (l > maxL) maxL = l; }
    const yPad = (maxL - minL) * .1 || 1, yMin = minL - yPad, yMax = maxL + yPad;
    const maxEp = lossHistory[lossHistory.length - 1].epoch;
    let drawTo = flowPhase === "training" ? flowEpochDisplay : maxEp;
    ctx.fillStyle = "#354555"; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
    for (let e = 0; e <= maxEp; e += 100) ctx.fillText(e, m.left + (e / maxEp) * pW, m.top + pH + 11);
    ctx.textAlign = "right"; for (let i = 0; i <= 3; i++) { const v = yMin + (yMax - yMin) * (1 - i / 3); ctx.fillText(v.toFixed(1), m.left - 4, m.top + (i / 3) * pH + 3); }
    ctx.strokeStyle = "#70c8a0"; ctx.lineWidth = 1.5; ctx.beginPath(); let started = false;
    for (const { epoch, loss } of lossHistory) { if (epoch > drawTo) break; if (!isFinite(loss)) continue; const x = m.left + (epoch / maxEp) * pW, y = m.top + ((yMax - loss) / (yMax - yMin)) * pH; if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); } ctx.stroke();
    if (flowPhase !== "idle" && flowPhase !== "training") { ctx.fillStyle = "#70c8a0"; ctx.font = "9px 'IBM Plex Mono',monospace"; ctx.textAlign = "right"; ctx.fillText(`final: ${losses[losses.length - 1].toFixed(2)}`, m.left + pW - 2, m.top + 9); }
  }, [lossHistory, flowPhase, flowEpochDisplay, computing]);

  // Marginal drawer
  function drawMarg(canvas, exact, label, trueV, range, samples, getV, hCol, lCol) {
    if (!canvas) return; const ctx = canvas.getContext("2d"); const cW = canvas.width, cH = canvas.height;
    const m = { top: 12, right: 14, bottom: 24, left: 40 }; const pW = cW - m.left - m.right, pH = cH - m.top - m.bottom;
    ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, cW, cH);
    const nB = 45, d = (range[1] - range[0]) / nB; const bins = new Float64Array(nB); let cnt = 0;
    for (const s of samples) { const v = getV(s); const b = Math.floor((v - range[0]) / d); if (b >= 0 && b < nB) { bins[b]++; cnt++; } }
    if (cnt > 0) for (let i = 0; i < nB; i++) bins[i] /= cnt * d;
    const dE = (range[1] - range[0]) / GRID_N; let mxE = 0; for (let i = 0; i < GRID_N; i++) if (exact[i] / dE > mxE) mxE = exact[i] / dE;
    let mxH = 0; for (let i = 0; i < nB; i++) if (bins[i] > mxH) mxH = bins[i]; const mxY = Math.max(mxE, mxH) * 1.15 || 1;
    ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
    ctx.fillStyle = "rgba(60,180,200,.08)"; ctx.strokeStyle = "rgba(60,180,200,.4)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(m.left, m.top + pH); for (let i = 0; i < GRID_N; i++) ctx.lineTo(m.left + ((i + .5) / GRID_N) * pW, m.top + pH - (exact[i] / dE / mxY) * pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.closePath(); ctx.fill();
    ctx.beginPath(); for (let i = 0; i < GRID_N; i++) { const x = m.left + ((i + .5) / GRID_N) * pW, y = m.top + pH - (exact[i] / dE / mxY) * pH; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
    if (cnt > 0) { const bW = pW / nB; ctx.fillStyle = hCol; for (let i = 0; i < nB; i++) { if (bins[i] === 0) continue; ctx.fillRect(m.left + i * bW, m.top + pH - (bins[i] / mxY) * pH, bW - .5, (bins[i] / mxY) * pH); } }
    if (trueV != null) { const tX = m.left + ((trueV - range[0]) / (range[1] - range[0])) * pW; ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(tX, m.top); ctx.lineTo(tX, m.top + pH); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = "#506878"; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText(label, m.left + pW / 2, cH - 3);
    ctx.textAlign = "left"; ctx.fillStyle = lCol; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.fillText(`p(${label}|d)`, m.left + 3, m.top + 9);
    ctx.fillStyle = "#354555"; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
    const step = niceStep(range[0], range[1]);
    for (let v = Math.ceil(range[0] / step) * step; v <= range[1]; v += step) ctx.fillText(fmtTick(v), m.left + ((v - range[0]) / (range[1] - range[0])) * pW, m.top + pH + 11);
  }

  useEffect(() => {
    const burn = 200, samples = [];
    for (let i = burn; i <= mcmcStep; i++) samples.push(fullChain[i]);
    drawMarg(mcmcM0Ref.current, ex0, dist.labels[0], dist.trueVals[0], dist.ranges[0], samples, s => s.p0, "rgba(255,140,70,.5)", "#e8a060");
    drawMarg(mcmcM1Ref.current, ex1, dist.labels[1], dist.trueVals[1], dist.ranges[1], samples, s => s.p1, "rgba(255,140,70,.5)", "#e8a060");
  }, [mcmcFrame, ex0, ex1, distKey, fullChain]);

  useEffect(() => {
    drawMarg(flowM0Ref.current, ex0, dist.labels[0], dist.trueVals[0], dist.ranges[0], flowSamples, s => s[0], "rgba(100,210,130,.5)", "#70c8a0");
    drawMarg(flowM1Ref.current, ex1, dist.labels[1], dist.trueVals[1], dist.ranges[1], flowSamples, s => s[1], "rgba(100,210,130,.5)", "#70c8a0");
  }, [flowSamples, ex0, ex1, distKey]);

  function handleFlowAction() {
    if (flowPhase === "idle" && !computing) { setFlowPhase("training"); setFlowSamples([]); }
    else if (flowPhase === "trained") setFlowPhase("sampling");
    else if (flowPhase === "sampling") { setFlowPhase("idle"); setFlowSamples([]); }
  }

  const flowBtnLabel = { idle: "Train ▸", training: "Training...", trained: "Sample ▸", sampling: "↺ Reset" }[flowPhase] || "Train ▸";

  return (
    <div style={{ background: "#08081a", minHeight: "100vh", display: "flex", fontFamily: "'IBM Plex Mono','SF Mono',monospace", color: "#c8d8e8" }}>
      {/* Sidebar */}
      <div style={{ width: SIDE_W, minWidth: SIDE_W, padding: "16px 10px", borderRight: "1px solid #1a2a38", display: "flex", flexDirection: "column", gap: "4px" }}>
        <div style={{ fontSize: "11px", color: "#70a0b8", fontWeight: 600, letterSpacing: ".05em", marginBottom: "6px" }}>DISTRIBUTION</div>
        {Object.entries(DISTRIBUTIONS).map(([key, d]) => (
          <button key={key} onClick={() => setDistKey(key)}
            style={{
              background: distKey === key ? "#12203a" : "transparent",
              border: `1px solid ${distKey === key ? "#3898b0" : "#1a2a38"}`,
              borderRadius: "5px", padding: "8px 9px", cursor: "pointer",
              textAlign: "left", transition: "all .15s",
            }}>
            <div style={{ fontSize: "11px", color: distKey === key ? "#a0e8ff" : "#7090a0", fontWeight: distKey === key ? 600 : 400 }}>{d.name}</div>
            <div style={{ fontSize: "9px", color: "#455565", marginTop: "2px", lineHeight: 1.3 }}>{d.detail}</div>
          </button>
        ))}

        <div style={{ marginTop: "14px", fontSize: "11px", color: "#70a0b8", fontWeight: 600, letterSpacing: ".05em", marginBottom: "4px" }}>FLOW LAYERS</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
          {[1, 2, 3, 5, 8, 12, 20].map(n => (
            <button key={n} onClick={() => setNLayers(n)} disabled={computing || flowPhase === "training"}
              style={{ ...smallBtn, borderColor: nLayers === n ? "#70c8a0" : "#253545", color: nLayers === n ? "#a0f0c8" : "#607888", fontWeight: nLayers === n ? 700 : 400 }}>
              {n}
            </button>
          ))}
        </div>
        <div style={{ fontSize: "9px", color: "#455565", marginTop: "2px" }}>{nLayers * 5} parameters</div>

        <div style={{ marginTop: "auto", fontSize: "9px", color: "#303848", lineHeight: 1.6 }}>
          <span style={{ color: "rgba(60,180,200,.5)" }}>—</span> exact<br />
          <span style={{ color: "rgba(255,140,70,.7)" }}>█</span> MCMC<br />
          <span style={{ color: "rgba(100,210,130,.7)" }}>█</span> flow<br />
          <span style={{ color: "rgba(255,255,255,.25)" }}>┊</span> true value
        </div>
      </div>

      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", padding: "12px 10px", overflow: "auto" }}>
        <h1 style={{ fontSize: "14px", fontWeight: 600, letterSpacing: ".05em", color: "#e0f0ff", margin: "0 0 1px" }}>MCMC vs Normalizing Flow</h1>
        <p style={{ fontSize: "10px", color: "#506878", margin: "0 0 8px" }}>{dist.desc}</p>

        {/* Two columns */}
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", justifyContent: "center", marginBottom: "4px" }}>
          {/* MCMC */}
          <div style={{ display: "flex", flexDirection: "column", gap: "3px", alignItems: "center" }}>
            <canvas ref={mcmc2dRef} width={COL_W} height={P2D_H} style={cs} />
            <canvas ref={mcmcM0Ref} width={COL_W} height={MH} style={cs} />
            <canvas ref={mcmcM1Ref} width={COL_W} height={MH} style={cs} />
            <div style={{ display: "flex", alignItems: "center", gap: "6px", width: COL_W, marginTop: "1px" }}>
              <input type="range" min={0} max={MCMC_FRAMES} step={1} value={mcmcFrame}
                onChange={e => { setMcmcFrame(parseInt(e.target.value)); setMcmcPlaying(false); }}
                style={{ flex: 1, accentColor: "#e8a060" }} />
            </div>
            <div style={{ display: "flex", gap: "4px" }}>
              <button onClick={() => { if (mcmcFrame >= MCMC_FRAMES) setMcmcFrame(0); setMcmcPlaying(p => !p); }} style={ctrlBtn}>{mcmcPlaying ? "⏸" : "▶"}</button>
              <button onClick={() => { setMcmcPlaying(false); setMcmcFrame(0); }} style={ctrlBtn}>↺</button>
              {[1, 3, 6].map(s => <button key={s} onClick={() => setMcmcSpeed(s)} style={{ ...ctrlBtn, borderColor: mcmcSpeed === s ? "#e8a060" : "#253545" }}>{s}×</button>)}
              <span style={{ fontSize: "9px", color: "#605848", alignSelf: "center", marginLeft: "2px" }}>sampling</span>
            </div>
          </div>

          {/* Flow */}
          <div style={{ display: "flex", flexDirection: "column", gap: "3px", alignItems: "center" }}>
            <canvas ref={flow2dRef} width={COL_W} height={P2D_H} style={cs} />
            <canvas ref={flowM0Ref} width={COL_W} height={MH} style={cs} />
            <canvas ref={flowM1Ref} width={COL_W} height={MH} style={cs} />
            <div style={{ display: "flex", gap: "4px", marginTop: "1px", alignItems: "center" }}>
              <button onClick={handleFlowAction} disabled={computing || flowPhase === "training"}
                style={{ ...ctrlBtn, borderColor: "#70c8a0", color: (computing || flowPhase === "training") ? "#354555" : "#a0f0c8", minWidth: 78, fontWeight: 600 }}>
                {computing ? `${trainProgress ?? 0}%` : flowBtnLabel}
              </button>
              {flowPhase !== "idle" && ["Train", "Sample"].map((lbl, i) => {
                const active = (i === 0 && flowPhase === "training") || (i === 1 && flowPhase === "sampling");
                const done = (i === 0 && flowPhase !== "idle") || (i === 1 && flowPhase === "sampling");
                const col = i === 0 ? "#c0a050" : "#70c8a0";
                return <span key={lbl} style={{ fontSize: "8px", padding: "2px 5px", borderRadius: "3px", background: active ? col + "22" : "transparent", color: done || active ? col : "#354555", border: `1px solid ${active ? col : "transparent"}` }}>{i + 1} {lbl}</span>;
              })}
            </div>
          </div>
        </div>

        {/* Loss */}
        <canvas ref={lossRef} width={COL_W * 2 + 8} height={LOSS_H} style={{ ...cs, maxWidth: "95vw", marginTop: "4px" }} />
      </div>
    </div>
  );
}

// ============================================================
// Helpers
// ============================================================
function niceStep(lo, hi) {
  const range = hi - lo;
  if (range <= 4) return 1;
  if (range <= 8) return 2;
  if (range <= 16) return 4;
  return 5;
}
function fmtTick(v) { return Math.abs(v - Math.round(v)) < 0.01 ? String(Math.round(v)) : v.toFixed(1); }

const cs = { borderRadius: "4px", border: "1px solid #1a2a38" };
const ctrlBtn = { background: "#0e1a28", border: "1px solid #253545", borderRadius: "4px", color: "#90c8e0", padding: "3px 8px", fontSize: "10px", cursor: "pointer", fontFamily: "inherit" };
const smallBtn = { background: "#0e1a28", border: "1px solid #253545", borderRadius: "3px", color: "#90c8e0", padding: "3px 8px", fontSize: "10px", cursor: "pointer", fontFamily: "inherit" };
