import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ============================================================
// PRNG
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
// Distributions
// ============================================================
const SIGMA_OBS = 0.8, N_DATA = 8;
const rngData = mulberry32(77);
const tObs = Array.from({ length: N_DATA }, (_, i) => (i + 0.5) / N_DATA);
const dataObs = tObs.map(t => 2.2 * Math.sin(2 * Math.PI * t + 1.8) + SIGMA_OBS * seededGaussian(rngData));

const DISTRIBUTIONS = {
  sinusoid: {
    name: "Sinusoid",
    desc: "A sin(2πt+φ) + noise",
    detail: "Curved ridge from amplitude–phase degeneracy",
    labels: ["A", "φ"], ranges: [[0.2, 4.5], [0, 2 * Math.PI]], trueVals: [2.2, 1.8],
    logP(p0, p1) {
      if (p0 < 0.2 || p0 > 4.5 || p1 < 0 || p1 > 2 * Math.PI) return -Infinity;
      let ll = 0;
      for (let i = 0; i < N_DATA; i++) { const r = dataObs[i] - p0 * Math.sin(2 * Math.PI * tObs[i] + p1); ll -= r * r / (2 * SIGMA_OBS * SIGMA_OBS); }
      return ll;
    }
  },
  banana: {
    name: "Banana",
    desc: "Rosenbrock-like",
    detail: "Strong nonlinear correlation along a curved ridge",
    labels: ["x₁", "x₂"], ranges: [[-2.5, 4.5], [-2, 12]], trueVals: [1.0, 1.0],
    logP(x, y) {
      if (x < -2.5 || x > 4.5 || y < -2 || y > 12) return -Infinity;
      return -((x - 1) ** 2) / 2 - 4 * ((y - x * x) ** 2);
    }
  },
  triplemode: {
    name: "Triple mode",
    desc: "Three overlapping Gaussians",
    detail: "Asymmetric modes with different orientations",
    labels: ["x₁", "x₂"], ranges: [[-4, 6], [-4, 6]], trueVals: [null, null],
    logP(x, y) {
      if (x < -4 || x > 6 || y < -4 || y > 6) return -Infinity;
      const dx1 = x, dy1 = y;
      const g1 = Math.exp(-0.5 * (1.2 * dx1 * dx1 - 0.8 * dx1 * dy1 + 1.2 * dy1 * dy1));
      const dx2 = x - 2.5, dy2 = y - 2.5;
      const g2 = 0.8 * Math.exp(-0.5 * (1.5 * dx2 * dx2 + 1.0 * dx2 * dy2 + 1.5 * dy2 * dy2));
      const dx3 = x - 1, dy3 = y + 1.5;
      const g3 = 0.6 * Math.exp(-0.5 * (0.6 * dx3 * dx3 + 0.6 * dy3 * dy3));
      return Math.log(g1 + g2 + g3 + 1e-30);
    }
  },
  ring: {
    name: "Ring",
    desc: "Annular posterior (r ≈ 3)",
    detail: "Topological challenge: flow must evacuate the interior",
    labels: ["x₁", "x₂"], ranges: [[-5.5, 5.5], [-5.5, 5.5]], trueVals: [null, null],
    logP(x, y) {
      if (x < -5.5 || x > 5.5 || y < -5.5 || y > 5.5) return -Infinity;
      const r = Math.sqrt(x * x + y * y);
      return -((r - 3) ** 2) / (2 * 0.35 * 0.35);
    }
  },
  funnel: {
    name: "Funnel",
    desc: "Neal's funnel",
    detail: "Hierarchical pathology: narrow neck + wide base",
    labels: ["x₁", "x₂"], ranges: [[-8, 8], [-4, 4]], trueVals: [null, null],
    logP(x, v) {
      if (x < -8 || x > 8 || v < -4 || v > 4) return -Infinity;
      const logpv = -(v * v) / (2 * 9);
      const sigma2 = Math.exp(v);
      const logpx = -0.5 * Math.log(2 * Math.PI * sigma2) - (x * x) / (2 * sigma2);
      return logpv + logpx;
    }
  },
};

// ============================================================
// Exact posterior grid
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
function computeMarginals(grid) {
  const m0 = new Float64Array(GRID_N), m1 = new Float64Array(GRID_N);
  for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) { m0[i] += grid[j * GRID_N + i]; m1[j] += grid[j * GRID_N + i]; }
  let s0 = 0, s1 = 0;
  for (let i = 0; i < GRID_N; i++) { s0 += m0[i]; s1 += m1[i]; }
  for (let i = 0; i < GRID_N; i++) { m0[i] /= s0; m1[i] /= s1; }
  return [m0, m1];
}

// ============================================================
// Flow architectures
// ============================================================

// --- Shared helpers ---
function flowToPhysical(x, ranges) {
  return [
    ranges[0][0] + (ranges[0][1] - ranges[0][0]) / (1 + Math.exp(-x[0])),
    ranges[1][0] + (ranges[1][1] - ranges[1][0]) / (1 + Math.exp(-x[1])),
  ];
}
function logSigJac(xRaw, rng) {
  const s = 1 / (1 + Math.exp(-xRaw));
  return Math.log(s * (1 - s) + 1e-10) + Math.log(rng[1] - rng[0]);
}

// --- Planar ---
function planarInit(nL, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: nL }, () => ({
    type: "planar",
    u: [.3 * seededGaussian(rng), .3 * seededGaussian(rng)],
    w: [.3 * seededGaussian(rng), .3 * seededGaussian(rng)],
    b: .1 * seededGaussian(rng),
  }));
}
function planarForward(z, params) {
  let x = [...z], ld = 0;
  for (const { u, w, b } of params) {
    const dot = w[0] * x[0] + w[1] * x[1] + b;
    const th = Math.tanh(dot), dth = 1 - th * th;
    ld += Math.log(Math.abs(1 + u[0] * dth * w[0] + u[1] * dth * w[1]) + 1e-10);
    x = [x[0] + u[0] * th, x[1] + u[1] * th];
  }
  return { x, logDet: ld };
}

// --- Radial ---
function radialInit(nL, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: nL }, () => ({
    type: "radial",
    z0: [.3 * seededGaussian(rng), .3 * seededGaussian(rng)],
    logAlpha: .1 * seededGaussian(rng),
    beta: .2 * seededGaussian(rng),
  }));
}
function radialForward(z, params) {
  let x = [...z], ld = 0;
  for (const { z0, logAlpha, beta } of params) {
    const alpha = Math.exp(logAlpha) + 0.01; // ensure positive
    const dx = x[0] - z0[0], dy = x[1] - z0[1];
    const r = Math.sqrt(dx * dx + dy * dy) + 1e-8;
    const h = 1 / (alpha + r);
    const hp = -h * h; // dh/dr
    const scale = 1 + beta * h;
    const dscale = beta * hp;
    // det J = scale * (scale + r * dscale) for d=2
    const det = scale * (scale + r * dscale);
    ld += Math.log(Math.abs(det) + 1e-10);
    x = [x[0] + beta * h * dx, x[1] + beta * h * dy];
  }
  return { x, logDet: ld };
}

// --- Affine Coupling ---
// Each layer: one dim is identity, other gets scale+shift conditioned on the first
// Conditioner: s(z) = a1 * tanh(a2 * z + a3) + a4, t(z) = b1 * tanh(b2 * z + b3) + b4
function couplingInit(nL, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: nL }, (_, i) => ({
    type: "coupling",
    // Alternate which dimension is the identity
    dim: i % 2,
    sa1: .3 * seededGaussian(rng), sa2: .5 + .2 * seededGaussian(rng),
    sa3: .1 * seededGaussian(rng), sa4: .1 * seededGaussian(rng),
    tb1: .3 * seededGaussian(rng), tb2: .5 + .2 * seededGaussian(rng),
    tb3: .1 * seededGaussian(rng), tb4: .1 * seededGaussian(rng),
  }));
}
function couplingForward(z, params) {
  let x = [...z], ld = 0;
  for (const L of params) {
    const d = L.dim; // which dim is identity
    const other = 1 - d;
    const cond = x[d]; // conditioning variable
    const logS = L.sa1 * Math.tanh(L.sa2 * cond + L.sa3) + L.sa4;
    const t = L.tb1 * Math.tanh(L.tb2 * cond + L.tb3) + L.tb4;
    x[other] = x[other] * Math.exp(logS) + t;
    ld += logS; // log |det J| for this layer
  }
  return { x, logDet: ld };
}

// ============================================================
// Generic training
// ============================================================
function genericForward(z, params) {
  const type = params[0]?.type;
  if (type === "planar") return planarForward(z, params);
  if (type === "radial") return radialForward(z, params);
  if (type === "coupling") return couplingForward(z, params);
  return planarForward(z, params);
}

function deepClone(params) { return JSON.parse(JSON.stringify(params)); }

function flattenParams(params) {
  const vals = [];
  for (const L of params) {
    if (L.type === "planar") { vals.push(L.u[0], L.u[1], L.w[0], L.w[1], L.b); }
    else if (L.type === "radial") { vals.push(L.z0[0], L.z0[1], L.logAlpha, L.beta); }
    else if (L.type === "coupling") { vals.push(L.sa1, L.sa2, L.sa3, L.sa4, L.tb1, L.tb2, L.tb3, L.tb4); }
  }
  return vals;
}

function unflattenParams(vals, template) {
  const out = deepClone(template);
  let idx = 0;
  for (const L of out) {
    if (L.type === "planar") { L.u[0] = vals[idx++]; L.u[1] = vals[idx++]; L.w[0] = vals[idx++]; L.w[1] = vals[idx++]; L.b = vals[idx++]; }
    else if (L.type === "radial") { L.z0[0] = vals[idx++]; L.z0[1] = vals[idx++]; L.logAlpha = vals[idx++]; L.beta = vals[idx++]; }
    else if (L.type === "coupling") { L.sa1 = vals[idx++]; L.sa2 = vals[idx++]; L.sa3 = vals[idx++]; L.sa4 = vals[idx++]; L.tb1 = vals[idx++]; L.tb2 = vals[idx++]; L.tb3 = vals[idx++]; L.tb4 = vals[idx++]; }
  }
  return out;
}

function trainStepGeneric(params, bs, lr, rng, dist) {
  const eps = 1e-4;
  const zBatch = Array.from({ length: bs }, () => [seededGaussian(rng), seededGaussian(rng)]);

  function loss(p) {
    let total = 0;
    for (const z of zBatch) {
      const { x, logDet } = genericForward(z, p);
      const phys = flowToPhysical(x, dist.ranges);
      total += -logDet - logSigJac(x[0], dist.ranges[0]) - logSigJac(x[1], dist.ranges[1]) - dist.logP(phys[0], phys[1]);
    }
    return total / bs;
  }

  const flat = flattenParams(params);
  const nP = flat.length;
  const grad = new Float64Array(nP);
  const baseLoss = loss(params);

  for (let i = 0; i < nP; i++) {
    flat[i] += eps;
    const lp = loss(unflattenParams(flat, params));
    flat[i] -= 2 * eps;
    const lm = loss(unflattenParams(flat, params));
    flat[i] += eps;
    grad[i] = Math.max(-2, Math.min(2, (lp - lm) / (2 * eps)));
  }

  for (let i = 0; i < nP; i++) flat[i] -= lr * grad[i];

  return { params: unflattenParams(flat, params), loss: baseLoss };
}

function sampleFlowGeneric(params, n, seed, dist) {
  const rng = mulberry32(seed), out = [];
  const [r0, r1] = dist.ranges;
  for (let i = 0; i < n; i++) {
    const z = [seededGaussian(rng), seededGaussian(rng)];
    const { x } = genericForward(z, params);
    const phys = flowToPhysical(x, dist.ranges);
    if (phys[0] >= r0[0] && phys[0] <= r0[1] && phys[1] >= r1[0] && phys[1] <= r1[1]) out.push(phys);
  }
  return out;
}

// ============================================================
// Async precompute
// ============================================================
const TOTAL_EPOCHS = 3000, SNAP_INT = 18;

function precomputeAsync(initFn, nL, seed, dist, onProgress, onDone) {
  let p = initFn(nL, seed);
  const rng = mulberry32(999);
  const snaps = [{ epoch: 0, params: deepClone(p), loss: null }];
  const lossHist = [];
  const nParams = flattenParams(p).length;
  const bs = nParams > 80 ? 60 : 100;
  let e = 1;
  const CHUNK = nParams > 80 ? 8 : 15;

  function step() {
    const end = Math.min(e + CHUNK, TOTAL_EPOCHS + 1);
    for (; e < end; e++) {
      const r = trainStepGeneric(p, bs, .02 * Math.exp(-e * .0008), rng, dist);
      p = r.params;
      lossHist.push({ epoch: e, loss: r.loss });
      if (e % SNAP_INT === 0 || e === TOTAL_EPOCHS)
        snaps.push({ epoch: e, params: deepClone(p), loss: r.loss });
    }
    if (e <= TOTAL_EPOCHS) { onProgress(e, TOTAL_EPOCHS); requestAnimationFrame(step); }
    else onDone(snaps, lossHist);
  }
  requestAnimationFrame(step);
}

// ============================================================
// Viridis
// ============================================================
const VIR = [[68,1,84],[72,34,115],[64,67,135],[52,94,141],[41,120,142],[32,144,140],[34,167,132],[68,190,112],[121,209,81],[189,222,38],[253,231,37]];
function viridis(t) { t = Math.max(0, Math.min(1, t)); const idx = t * (VIR.length - 1), lo = Math.floor(idx), hi = Math.min(lo + 1, VIR.length - 1), f = idx - lo; return VIR[lo].map((c, i) => Math.round(c + f * (VIR[hi][i] - c))); }

// ============================================================
// Layout
// ============================================================
const COL_W = 260, P2D_H = 230, MH = 90, LOSS_H = 100;
const PAD = { top: 24, right: 10, bottom: 26, left: 36 };
function plotArea(w, h) { return { x: PAD.left, y: PAD.top, w: w - PAD.left - PAD.right, h: h - PAD.top - PAD.bottom }; }

// ============================================================
// Flow type config
// ============================================================
const FLOW_TYPES = [
  { key: "planar", name: "Planar", color: "#e8a060", initFn: planarInit, pPerLayer: 5, desc: "z' = z + u tanh(wᵀz + b)" },
  { key: "radial", name: "Radial", color: "#c080e0", initFn: radialInit, pPerLayer: 4, desc: "z' = z + βh(α,r)(z−z₀)" },
  { key: "coupling", name: "Coupling", color: "#70c8a0", initFn: couplingInit, pPerLayer: 8, desc: "x_b = z_b·eˢ⁽ᶻᵃ⁾ + t(zₐ)" },
];

// ============================================================
// Single flow column component
// ============================================================
function FlowColumn({ flowType, dist, nLayers, pGrid, exMargs, bgImg, distKey }) {
  const ref2d = useRef(null), refM0 = useRef(null), refM1 = useRef(null), refLoss = useRef(null);
  const [phase, setPhase] = useState("idle"); // idle | precomputing | ready | training | trained | sampling
  const [progress, setProgress] = useState(0);
  const [epochDisplay, setEpochDisplay] = useState(0);
  const [samples, setSamples] = useState([]);
  const [snaps, setSnaps] = useState(null);
  const [lossHist, setLossHist] = useState([]);
  const [trainedParams, setTrainedParams] = useState(null);
  const genRef = useRef(0);
  const animRef = useRef(null);

  // Precompute on mount or when dist/layers change
  useEffect(() => {
    setPhase("precomputing"); setProgress(0); setSamples([]); setEpochDisplay(0);
    setSnaps(null); setLossHist([]); setTrainedParams(null);
    const gen = ++genRef.current;
    precomputeAsync(flowType.initFn, nLayers, 55, dist,
      (e, t) => { if (gen === genRef.current) setProgress(Math.round(e / t * 100)); },
      (s, lh) => { if (gen === genRef.current) { setSnaps(s); setLossHist(lh); setPhase("ready"); setProgress(100); } }
    );
    return () => { genRef.current++; if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [distKey, nLayers, flowType.key]);

  // Training animation
  useEffect(() => {
    if (phase !== "training" || !snaps) { if (animRef.current) cancelAnimationFrame(animRef.current); return; }
    let idx = 0;
    const tick = () => {
      idx += 2;
      if (idx >= snaps.length) {
        const fin = snaps[snaps.length - 1];
        setEpochDisplay(fin.epoch); setTrainedParams(fin.params); setPhase("trained"); return;
      }
      setEpochDisplay(snaps[idx].epoch);
      setSamples(sampleFlowGeneric(snaps[idx].params, 800, 42, dist));
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [phase, snaps]);

  // Sampling animation
  useEffect(() => {
    if (phase !== "sampling" || !trainedParams) return;
    let count = 0, acc = [];
    const tick = () => {
      const rng = mulberry32(42 + count);
      for (let i = 0; i < 100; i++) {
        const z = [seededGaussian(rng), seededGaussian(rng)];
        const { x } = genericForward(z, trainedParams);
        const phys = flowToPhysical(x, dist.ranges);
        if (phys[0] >= dist.ranges[0][0] && phys[0] <= dist.ranges[0][1] && phys[1] >= dist.ranges[1][0] && phys[1] <= dist.ranges[1][1]) acc.push(phys);
      }
      count += 100; setSamples([...acc]);
      if (acc.length < 1500) animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [phase, trainedParams]);

  const toPlot = useCallback((v0, v1, p) => [
    p.x + ((v0 - dist.ranges[0][0]) / (dist.ranges[0][1] - dist.ranges[0][0])) * p.w,
    p.y + ((v1 - dist.ranges[1][0]) / (dist.ranges[1][1] - dist.ranges[1][0])) * p.h
  ], [dist]);

  // Draw 2D
  useEffect(() => {
    const c = ref2d.current; if (!c) return;
    const ctx = c.getContext("2d"); const p = plotArea(COL_W, P2D_H);
    ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, COL_W, P2D_H);
    if (bgImg) { ctx.globalAlpha = .3; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high"; ctx.drawImage(bgImg, p.x, p.y, p.w, p.h); ctx.globalAlpha = 1; }
    ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x, p.y + p.h); ctx.lineTo(p.x + p.w, p.y + p.h); ctx.stroke();
    // Axis labels
    ctx.fillStyle = "#405565"; ctx.font = "9px 'IBM Plex Mono',monospace"; ctx.textAlign = "center"; ctx.fillText(dist.labels[0], p.x + p.w / 2, P2D_H - 3);
    ctx.save(); ctx.translate(9, p.y + p.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(dist.labels[1], 0, 0); ctx.restore();
    // Title
    ctx.fillStyle = flowType.color; ctx.font = "bold 10px 'IBM Plex Mono',monospace"; ctx.textAlign = "left"; ctx.fillText(flowType.name, p.x + 2, p.y - 8);
    ctx.fillStyle = "#405565"; ctx.font = "8px 'IBM Plex Mono',monospace";
    ctx.fillText(`${nLayers}L · ${nLayers * flowType.pPerLayer}p`, p.x + 2 + ctx.measureText(flowType.name).width + 6, p.y - 8);
    // Ticks
    ctx.fillStyle = "#2a3a4a"; ctx.font = "7px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
    const s0 = niceStep(dist.ranges[0][0], dist.ranges[0][1]);
    for (let v = Math.ceil(dist.ranges[0][0] / s0) * s0; v <= dist.ranges[0][1]; v += s0) { const [x] = toPlot(v, 0, p); ctx.fillText(fmtTick(v), x, p.y + p.h + 9); }
    ctx.textAlign = "right";
    const s1 = niceStep(dist.ranges[1][0], dist.ranges[1][1]);
    for (let v = Math.ceil(dist.ranges[1][0] / s1) * s1; v <= dist.ranges[1][1]; v += s1) { const [, y] = toPlot(0, v, p); ctx.fillText(fmtTick(v), p.x - 3, y + 3); }

    // State-dependent content
    if (phase === "precomputing") {
      ctx.fillStyle = "rgba(12,12,32,.7)"; ctx.fillRect(p.x, p.y, p.w, p.h);
      ctx.fillStyle = flowType.color; ctx.font = "11px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText(`Precomputing ${progress}%`, p.x + p.w / 2, p.y + p.h / 2);
    } else if (phase === "ready" || phase === "idle") {
      ctx.fillStyle = "rgba(12,12,32,.5)"; ctx.fillRect(p.x, p.y, p.w, p.h);
      ctx.fillStyle = "#506878"; ctx.font = "10px 'IBM Plex Mono',monospace"; ctx.textAlign = "center";
      ctx.fillText("Train ▸", p.x + p.w / 2, p.y + p.h / 2);
    } else {
      const alpha = phase === "training" ? .2 : .3;
      ctx.fillStyle = flowType.color.slice(0, 7) + (phase === "training" ? "33" : "55");
      for (const s of samples) { const [x, y] = toPlot(s[0], s[1], p); ctx.beginPath(); ctx.arc(x, y, 1.3, 0, 2 * Math.PI); ctx.fill(); }
      ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "right"; ctx.fillStyle = flowType.color + "aa";
      if (phase === "training") ctx.fillText(`epoch ${epochDisplay}`, p.x + p.w - 2, p.y + 9);
      else if (phase === "trained") ctx.fillText("trained ✓", p.x + p.w - 2, p.y + 9);
      else if (phase === "sampling") { ctx.fillText(`n=${samples.length}`, p.x + p.w - 2, p.y + 9); ctx.fillText(`ESS=${samples.length}`, p.x + p.w - 2, p.y + 19); }
    }
  }, [phase, samples, epochDisplay, progress, bgImg, dist, flowType, nLayers, toPlot]);

  // Marginals
  useEffect(() => {
    drawMarg(refM0.current, exMargs[0], dist.labels[0], dist.trueVals[0], dist.ranges[0], samples, s => s[0], flowType.color + "88", flowType.color);
    drawMarg(refM1.current, exMargs[1], dist.labels[1], dist.trueVals[1], dist.ranges[1], samples, s => s[1], flowType.color + "88", flowType.color);
  }, [samples, exMargs, dist, flowType]);

  // Loss curve
  useEffect(() => {
    const c = refLoss.current; if (!c) return;
    const ctx = c.getContext("2d"); const cW = c.width, cH = c.height;
    const m = { top: 10, right: 8, bottom: 16, left: 40 }; const pW = cW - m.left - m.right, pH = cH - m.top - m.bottom;
    ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, cW, cH);
    ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
    ctx.fillStyle = flowType.color; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "left"; ctx.fillText("loss", m.left + 2, m.top + 7);
    if (lossHist.length < 2) return;
    const losses = lossHist.map(l => l.loss).filter(l => isFinite(l)); if (losses.length < 2) return;
    let minL = Infinity, maxL = -Infinity; for (const l of losses) { if (l < minL) minL = l; if (l > maxL) maxL = l; }
    const yPad = (maxL - minL) * .1 || 1, yMin = minL - yPad, yMax = maxL + yPad;
    const maxEp = lossHist[lossHist.length - 1].epoch;
    const drawTo = phase === "training" ? epochDisplay : maxEp;
    ctx.strokeStyle = flowType.color; ctx.lineWidth = 1.2; ctx.beginPath(); let started = false;
    for (const { epoch, loss } of lossHist) { if (epoch > drawTo) break; if (!isFinite(loss)) continue; const x = m.left + (epoch / maxEp) * pW, y = m.top + ((yMax - loss) / (yMax - yMin)) * pH; if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y); } ctx.stroke();
    if (phase === "trained" || phase === "sampling") { ctx.fillStyle = flowType.color; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "right"; ctx.fillText(losses[losses.length - 1].toFixed(1), m.left + pW - 2, m.top + 7); }
  }, [lossHist, phase, epochDisplay, flowType]);

  function handleAction() {
    if (phase === "ready") { setPhase("training"); setSamples([]); }
    else if (phase === "trained") setPhase("sampling");
    else if (phase === "sampling") { setPhase("ready"); setSamples([]); }
  }

  const btnLabel = { idle: "...", precomputing: `${progress}%`, ready: "Train ▸", training: "Training...", trained: "Sample ▸", sampling: "↺ Reset" }[phase];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "3px", alignItems: "center" }}>
      <canvas ref={ref2d} width={COL_W} height={P2D_H} style={cs} />
      <canvas ref={refM0} width={COL_W} height={MH} style={cs} />
      <canvas ref={refM1} width={COL_W} height={MH} style={cs} />
      <canvas ref={refLoss} width={COL_W} height={65} style={cs} />
      <button onClick={handleAction} disabled={phase === "precomputing" || phase === "training" || phase === "idle"}
        style={{ ...ctrlBtn, borderColor: flowType.color, color: (phase === "precomputing" || phase === "training") ? "#354555" : flowType.color, minWidth: 80, fontWeight: 600, fontSize: "10px" }}>
        {btnLabel}
      </button>
    </div>
  );
}

// ============================================================
// Marginal drawer (shared)
// ============================================================
function drawMarg(canvas, exact, label, trueV, range, samples, getV, hCol, lCol) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d"); const cW = canvas.width, cH = canvas.height;
  const m = { top: 10, right: 10, bottom: 20, left: 36 }; const pW = cW - m.left - m.right, pH = cH - m.top - m.bottom;
  ctx.fillStyle = "#0c0c20"; ctx.fillRect(0, 0, cW, cH);
  const nB = 40, d = (range[1] - range[0]) / nB; const bins = new Float64Array(nB); let cnt = 0;
  for (const s of samples) { const v = getV(s); const b = Math.floor((v - range[0]) / d); if (b >= 0 && b < nB) { bins[b]++; cnt++; } }
  if (cnt > 0) for (let i = 0; i < nB; i++) bins[i] /= cnt * d;
  const dE = (range[1] - range[0]) / GRID_N; let mxE = 0; for (let i = 0; i < GRID_N; i++) if (exact[i] / dE > mxE) mxE = exact[i] / dE;
  let mxH = 0; for (let i = 0; i < nB; i++) if (bins[i] > mxH) mxH = bins[i]; const mxY = Math.max(mxE, mxH) * 1.15 || 1;
  ctx.strokeStyle = "#2a3a4a"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(m.left, m.top); ctx.lineTo(m.left, m.top + pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.stroke();
  ctx.fillStyle = "rgba(60,180,200,.06)"; ctx.strokeStyle = "rgba(60,180,200,.35)"; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(m.left, m.top + pH); for (let i = 0; i < GRID_N; i++) ctx.lineTo(m.left + ((i + .5) / GRID_N) * pW, m.top + pH - (exact[i] / dE / mxY) * pH); ctx.lineTo(m.left + pW, m.top + pH); ctx.closePath(); ctx.fill();
  ctx.beginPath(); for (let i = 0; i < GRID_N; i++) { const x = m.left + ((i + .5) / GRID_N) * pW, y = m.top + pH - (exact[i] / dE / mxY) * pH; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
  if (cnt > 0) { const bW = pW / nB; ctx.fillStyle = hCol; for (let i = 0; i < nB; i++) { if (bins[i] === 0) continue; ctx.fillRect(m.left + i * bW, m.top + pH - (bins[i] / mxY) * pH, bW - .5, (bins[i] / mxY) * pH); } }
  if (trueV != null) { const tX = m.left + ((trueV - range[0]) / (range[1] - range[0])) * pW; ctx.strokeStyle = "rgba(255,255,255,.2)"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(tX, m.top); ctx.lineTo(tX, m.top + pH); ctx.stroke(); ctx.setLineDash([]); }
  ctx.fillStyle = lCol; ctx.font = "8px 'IBM Plex Mono',monospace"; ctx.textAlign = "left"; ctx.fillText(`p(${label}|d)`, m.left + 2, m.top + 7);
}

// ============================================================
// Main
// ============================================================
export default function FlowComparison() {
  const [distKey, setDistKey] = useState("sinusoid");
  const [nLayers, setNLayers] = useState(12);
  const dist = DISTRIBUTIONS[distKey];

  const pGrid = useMemo(() => computeGrid(dist), [distKey]);
  const exMargs = useMemo(() => computeMarginals(pGrid), [pGrid]);
  const [bgImg, setBgImg] = useState(null);

  useEffect(() => {
    const img = new ImageData(GRID_N, GRID_N); let mx = 0;
    for (let i = 0; i < pGrid.length; i++) if (pGrid[i] > mx) mx = pGrid[i];
    for (let j = 0; j < GRID_N; j++) for (let i = 0; i < GRID_N; i++) {
      const v = mx > 0 ? Math.pow(pGrid[j * GRID_N + i] / mx, .6) : 0; const [r, g, b] = viridis(v); const idx = (j * GRID_N + i) * 4;
      img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
    }
    createImageBitmap(img).then(bmp => { const c = document.createElement("canvas"); c.width = GRID_N; c.height = GRID_N; c.getContext("2d").drawImage(bmp, 0, 0); setBgImg(c); });
  }, [pGrid]);

  return (
    <div style={{ background: "#08081a", minHeight: "100vh", display: "flex", fontFamily: "'IBM Plex Mono','SF Mono',monospace", color: "#c8d8e8" }}>
      {/* Sidebar */}
      <div style={{ width: 140, minWidth: 140, padding: "14px 8px", borderRight: "1px solid #1a2a38", display: "flex", flexDirection: "column", gap: "3px", overflow: "auto" }}>
        <div style={{ fontSize: "10px", color: "#70a0b8", fontWeight: 600, letterSpacing: ".05em", marginBottom: "4px" }}>TARGET</div>
        {Object.entries(DISTRIBUTIONS).map(([key, d]) => (
          <button key={key} onClick={() => setDistKey(key)}
            style={{ background: distKey === key ? "#12203a" : "transparent", border: `1px solid ${distKey === key ? "#3898b0" : "#1a2a38"}`, borderRadius: "4px", padding: "6px 7px", cursor: "pointer", textAlign: "left" }}>
            <div style={{ fontSize: "10px", color: distKey === key ? "#a0e8ff" : "#6888a0", fontWeight: distKey === key ? 600 : 400 }}>{d.name}</div>
            <div style={{ fontSize: "8px", color: "#405060", marginTop: "1px", lineHeight: 1.3 }}>{d.detail}</div>
          </button>
        ))}
        <div style={{ marginTop: "12px", fontSize: "10px", color: "#70a0b8", fontWeight: 600, letterSpacing: ".05em", marginBottom: "3px" }}>LAYERS</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "2px" }}>
          {[3, 5, 8, 12, 20].map(n => (
            <button key={n} onClick={() => setNLayers(n)}
              style={{ ...smallBtn, borderColor: nLayers === n ? "#3898b0" : "#253545", color: nLayers === n ? "#a0e8ff" : "#607888", fontWeight: nLayers === n ? 700 : 400 }}>{n}</button>
          ))}
        </div>

        <div style={{ marginTop: "12px", fontSize: "9px", color: "#304050", lineHeight: 1.8 }}>
          <div style={{ marginBottom: "6px", fontSize: "10px", color: "#70a0b8", fontWeight: 600, letterSpacing: ".05em" }}>ARCHITECTURES</div>
          {FLOW_TYPES.map(f => (
            <div key={f.key} style={{ marginBottom: "6px" }}>
              <span style={{ color: f.color, fontWeight: 600 }}>{f.name}</span>
              <div style={{ fontSize: "8px", color: "#405060" }}>{f.desc}</div>
              <div style={{ fontSize: "8px", color: "#354555" }}>{f.pPerLayer}p/layer</div>
            </div>
          ))}
        </div>

        <div style={{ marginTop: "auto", fontSize: "8px", color: "#283040", lineHeight: 1.6 }}>
          <span style={{ color: "rgba(60,180,200,.5)" }}>—</span> exact posterior<br />
          <span style={{ color: "rgba(255,255,255,.2)" }}>┊</span> true value
        </div>
      </div>

      {/* Main: 3 columns */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", padding: "10px 6px", overflow: "auto" }}>
        <h1 style={{ fontSize: "13px", fontWeight: 600, letterSpacing: ".05em", color: "#e0f0ff", margin: "0 0 1px" }}>Flow Architecture Comparison</h1>
        <p style={{ fontSize: "9px", color: "#506878", margin: "0 0 8px" }}>{dist.desc} — same target, same layers, different architectures</p>

        <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", justifyContent: "center" }}>
          {FLOW_TYPES.map(ft => (
            <FlowColumn key={ft.key + distKey + nLayers} flowType={ft} dist={dist} nLayers={nLayers}
              pGrid={pGrid} exMargs={exMargs} bgImg={bgImg} distKey={distKey} />
          ))}
        </div>
      </div>
    </div>
  );
}

function niceStep(lo, hi) { const r = hi - lo; if (r <= 4) return 1; if (r <= 8) return 2; if (r <= 16) return 4; return 5; }
function fmtTick(v) { return Math.abs(v - Math.round(v)) < 0.01 ? String(Math.round(v)) : v.toFixed(1); }
const cs = { borderRadius: "3px", border: "1px solid #1a2a38" };
const ctrlBtn = { background: "#0e1a28", border: "1px solid #253545", borderRadius: "4px", color: "#90c8e0", padding: "3px 8px", fontSize: "10px", cursor: "pointer", fontFamily: "inherit" };
const smallBtn = { background: "#0e1a28", border: "1px solid #253545", borderRadius: "3px", color: "#90c8e0", padding: "2px 7px", fontSize: "10px", cursor: "pointer", fontFamily: "inherit" };
