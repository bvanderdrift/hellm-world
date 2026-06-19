/** THIS FILE IS AI GENERATED */
import tgpu, { d } from "typegpu";
import { builtin } from "typegpu/data";
import * as std from "typegpu/std";
import { gpuContext } from "../shared/gpu-context.ts";
import { createMatrix } from "../shared/matrices.ts";
import {
  createMatrixBuffer,
  dotProductKernel,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";
import { maybeDumpWgsl } from "../bench-harness.ts";

const rand = () => Math.random() * 2 - 1;

const arr = (length: number) => d.struct({ values: d.arrayOf(d.f32, length) });
const arrSlot = arr(0);

const bwCfg = d.struct({
  n: d.u32,
});

const bwLayout = tgpu.bindGroupLayout({
  x: { storage: arrSlot, access: "readonly" },
  y: { storage: arrSlot, access: "readonly" },
  out: { storage: arrSlot, access: "mutable" },
  cfg: { uniform: bwCfg },
});

const saxpyKernel = tgpu.computeFn({
  in: {
    globalId: builtin.globalInvocationId,
    numWorkgroups: builtin.numWorkgroups,
  },
  workgroupSize: [256],
})((input) => {
  "use gpu";

  const n = bwLayout.$.cfg.n;
  const stride = input.numWorkgroups.x * 256;
  const two = d.f32(2);

  for (let i = input.globalId.x; i < n; i += stride) {
    bwLayout.$.out.values[i]! =
      two * bwLayout.$.x.values[i]! + bwLayout.$.y.values[i]!;
  }
});

const saxpyRunner = gpuContext.createComputePipeline({ compute: saxpyKernel });

// ---------------------------------------------------------------------------
// Edge-safe optimized matmul, used as the "what a tuned kernel could actually
// reach on real shapes" ceiling. NOT the model path.
//
// Per-workgroup 64x64 output tile, BK=8 K-strip, 8x4 per-thread micro-tile
// (128 threads). Shared tiles are vec4f so the inner loop reads A/B with
// vectorized loads and accumulates with vec4 fma. Unlike a "lab" kernel this
// handles ARBITRARY M/N/K: the load phase zero-pads out-of-range elements into
// shared memory (keeping the hot loop branch-free) and the write phase guards
// every store. That edge handling is what makes the number realistic.
// ---------------------------------------------------------------------------
const optCfg = d.struct({ M: d.u32, N: d.u32, K: d.u32 });

const optLayout = tgpu.bindGroupLayout({
  a: { storage: arrSlot, access: "readonly" },
  b: { storage: arrSlot, access: "readonly" },
  c: { storage: arrSlot, access: "mutable" },
  cfg: { uniform: optCfg },
});

const OPT_BM = 64;
const OPT_BN = 64;
const OPT_BK = 8;
const OPT_TN = 4;
const OPT_WG = 128;
const OPT_NV = OPT_BN / 4; // 16 vec4 columns
const OPT_MQ = OPT_BM / 4; // 16 vec4 rows

const optTileA = tgpu.workgroupVar(d.arrayOf(d.vec4f, (OPT_BM * OPT_BK) / 4));
const optTileB = tgpu.workgroupVar(d.arrayOf(d.vec4f, (OPT_BK * OPT_BN) / 4));

const optMatmulKernel = tgpu.computeFn({
  in: { wgId: builtin.workgroupId, tid: builtin.localInvocationIndex },
  workgroupSize: [OPT_WG],
})((input) => {
  "use gpu";
  const a = optLayout.$.a;
  const b = optLayout.$.b;
  const c = optLayout.$.c;
  const M = optLayout.$.cfg.M;
  const N = optLayout.$.cfg.N;
  const K = optLayout.$.cfg.K;

  const blockRow = input.wgId.y * OPT_BM;
  const blockCol = input.wgId.x * OPT_BN;
  const threadRowQ = d.u32(input.tid / d.u32(OPT_BN / OPT_TN)) * 2;
  const threadColV = input.tid % d.u32(OPT_NV);

  let acc0 = d.vec4f();
  let acc1 = d.vec4f();
  let acc2 = d.vec4f();
  let acc3 = d.vec4f();
  let acc4 = d.vec4f();
  let acc5 = d.vec4f();
  let acc6 = d.vec4f();
  let acc7 = d.vec4f();

  for (let kt = d.u32(0); kt < K; kt += OPT_BK) {
    {
      const kk = d.u32(input.tid / d.u32(OPT_MQ));
      const mq = input.tid % d.u32(OPT_MQ);
      const rowBase = blockRow + mq * 4;
      const kcol = kt + kk;
      const kOk = kcol < K;
      let a0 = d.f32(0);
      let a1 = d.f32(0);
      let a2 = d.f32(0);
      let a3 = d.f32(0);
      if (kOk && rowBase + 0 < M) a0 = a.values[(rowBase + 0) * K + kcol]!;
      if (kOk && rowBase + 1 < M) a1 = a.values[(rowBase + 1) * K + kcol]!;
      if (kOk && rowBase + 2 < M) a2 = a.values[(rowBase + 2) * K + kcol]!;
      if (kOk && rowBase + 3 < M) a3 = a.values[(rowBase + 3) * K + kcol]!;
      optTileA.$[input.tid] = d.vec4f(a0, a1, a2, a3);
    }
    {
      const kk = d.u32(input.tid / d.u32(OPT_NV));
      const nv = input.tid % d.u32(OPT_NV);
      const brow = kt + kk;
      const bcol = blockCol + nv * 4;
      const kOk = brow < K;
      let b0 = d.f32(0);
      let b1 = d.f32(0);
      let b2 = d.f32(0);
      let b3 = d.f32(0);
      if (kOk && bcol + 0 < N) b0 = b.values[brow * N + bcol + 0]!;
      if (kOk && bcol + 1 < N) b1 = b.values[brow * N + bcol + 1]!;
      if (kOk && bcol + 2 < N) b2 = b.values[brow * N + bcol + 2]!;
      if (kOk && bcol + 3 < N) b3 = b.values[brow * N + bcol + 3]!;
      optTileB.$[input.tid] = d.vec4f(b0, b1, b2, b3);
    }
    std.workgroupBarrier();

    for (let kk = 0; kk < OPT_BK; kk++) {
      const bVec = optTileB.$[kk * OPT_NV + threadColV]!;
      const aLo = optTileA.$[kk * OPT_MQ + threadRowQ]!;
      const aHi = optTileA.$[kk * OPT_MQ + threadRowQ + 1]!;
      acc0 = std.fma(d.vec4f(aLo.x, aLo.x, aLo.x, aLo.x), bVec, acc0);
      acc1 = std.fma(d.vec4f(aLo.y, aLo.y, aLo.y, aLo.y), bVec, acc1);
      acc2 = std.fma(d.vec4f(aLo.z, aLo.z, aLo.z, aLo.z), bVec, acc2);
      acc3 = std.fma(d.vec4f(aLo.w, aLo.w, aLo.w, aLo.w), bVec, acc3);
      acc4 = std.fma(d.vec4f(aHi.x, aHi.x, aHi.x, aHi.x), bVec, acc4);
      acc5 = std.fma(d.vec4f(aHi.y, aHi.y, aHi.y, aHi.y), bVec, acc5);
      acc6 = std.fma(d.vec4f(aHi.z, aHi.z, aHi.z, aHi.z), bVec, acc6);
      acc7 = std.fma(d.vec4f(aHi.w, aHi.w, aHi.w, aHi.w), bVec, acc7);
    }
    std.workgroupBarrier();
  }

  const col = blockCol + threadColV * 4;
  const rowStart = blockRow + threadRowQ * 4;
  const c0Ok = col + 0 < N;
  const c1Ok = col + 1 < N;
  const c2Ok = col + 2 < N;
  const c3Ok = col + 3 < N;

  if (rowStart + 0 < M) {
    const base = (rowStart + 0) * N + col;
    if (c0Ok) c.values[base + 0] = acc0.x;
    if (c1Ok) c.values[base + 1] = acc0.y;
    if (c2Ok) c.values[base + 2] = acc0.z;
    if (c3Ok) c.values[base + 3] = acc0.w;
  }
  if (rowStart + 1 < M) {
    const base = (rowStart + 1) * N + col;
    if (c0Ok) c.values[base + 0] = acc1.x;
    if (c1Ok) c.values[base + 1] = acc1.y;
    if (c2Ok) c.values[base + 2] = acc1.z;
    if (c3Ok) c.values[base + 3] = acc1.w;
  }
  if (rowStart + 2 < M) {
    const base = (rowStart + 2) * N + col;
    if (c0Ok) c.values[base + 0] = acc2.x;
    if (c1Ok) c.values[base + 1] = acc2.y;
    if (c2Ok) c.values[base + 2] = acc2.z;
    if (c3Ok) c.values[base + 3] = acc2.w;
  }
  if (rowStart + 3 < M) {
    const base = (rowStart + 3) * N + col;
    if (c0Ok) c.values[base + 0] = acc3.x;
    if (c1Ok) c.values[base + 1] = acc3.y;
    if (c2Ok) c.values[base + 2] = acc3.z;
    if (c3Ok) c.values[base + 3] = acc3.w;
  }
  if (rowStart + 4 < M) {
    const base = (rowStart + 4) * N + col;
    if (c0Ok) c.values[base + 0] = acc4.x;
    if (c1Ok) c.values[base + 1] = acc4.y;
    if (c2Ok) c.values[base + 2] = acc4.z;
    if (c3Ok) c.values[base + 3] = acc4.w;
  }
  if (rowStart + 5 < M) {
    const base = (rowStart + 5) * N + col;
    if (c0Ok) c.values[base + 0] = acc5.x;
    if (c1Ok) c.values[base + 1] = acc5.y;
    if (c2Ok) c.values[base + 2] = acc5.z;
    if (c3Ok) c.values[base + 3] = acc5.w;
  }
  if (rowStart + 6 < M) {
    const base = (rowStart + 6) * N + col;
    if (c0Ok) c.values[base + 0] = acc6.x;
    if (c1Ok) c.values[base + 1] = acc6.y;
    if (c2Ok) c.values[base + 2] = acc6.z;
    if (c3Ok) c.values[base + 3] = acc6.w;
  }
  if (rowStart + 7 < M) {
    const base = (rowStart + 7) * N + col;
    if (c0Ok) c.values[base + 0] = acc7.x;
    if (c1Ok) c.values[base + 1] = acc7.y;
    if (c2Ok) c.values[base + 2] = acc7.z;
    if (c3Ok) c.values[base + 3] = acc7.w;
  }
});

const optMatmulRunner = gpuContext.createComputePipeline({
  compute: optMatmulKernel,
});

const flush = () => gpuContext.device.queue.onSubmittedWorkDone();

const storageBuffer = (length: number, fill: boolean) => {
  const builder = gpuContext.createBuffer(
    arr(length),
    fill ? { values: Array.from({ length }, rand) } : undefined,
  );
  return builder.$usage("storage");
};

const WARMUP = 5;
const ITERS = 20;

const timeDispatch = async (dispatch: () => void) => {
  for (let i = 0; i < WARMUP; i++) dispatch();
  await flush();

  const start = performance.now();
  for (let i = 0; i < ITERS; i++) dispatch();
  await flush();
  return (performance.now() - start) / ITERS;
};

const measureCurrent = async (m: number, k: number, n: number) => {
  const a = createMatrixBuffer(createMatrix(m, k, rand));
  const b = createMatrixBuffer(createMatrix(k, n, rand));
  const out = createMatrixBuffer(createMatrix(m, n));
  const ms = await timeDispatch(() => multiplyMatricesOnGPU(a, b, out));
  return (2 * m * n * k) / (ms / 1000) / 1e9;
};

const measureOpt = async (m: number, k: number, n: number) => {
  const a = storageBuffer(m * k, true);
  const b = storageBuffer(k * n, true);
  const c = storageBuffer(m * n, false);
  const cfg = gpuContext
    .createBuffer(optCfg, { M: m, N: n, K: k })
    .$usage("uniform");
  const params = gpuContext.createBindGroup(optLayout, { a, b, c, cfg });
  const dispatch = () =>
    optMatmulRunner
      .with(params)
      .dispatchWorkgroups(Math.ceil(n / OPT_BN), Math.ceil(m / OPT_BM));
  const ms = await timeDispatch(dispatch);
  return (2 * m * n * k) / (ms / 1000) / 1e9;
};

const cpuRef = (A: Float32Array, B: Float32Array, m: number, k: number, n: number) => {
  const C = new Float32Array(m * n);
  for (let i = 0; i < m; i++)
    for (let p = 0; p < k; p++) {
      const aip = A[i * k + p]!;
      for (let j = 0; j < n; j++) C[i * n + j]! += aip * B[p * n + j]!;
    }
  return C;
};

const verifyOpt = async (m: number, k: number, n: number) => {
  const A = Float32Array.from({ length: m * k }, rand);
  const B = Float32Array.from({ length: k * n }, rand);
  const a = gpuContext
    .createBuffer(arr(m * k), { values: Array.from(A) })
    .$usage("storage");
  const b = gpuContext
    .createBuffer(arr(k * n), { values: Array.from(B) })
    .$usage("storage");
  const c = storageBuffer(m * n, false);
  const cfg = gpuContext
    .createBuffer(optCfg, { M: m, N: n, K: k })
    .$usage("uniform");
  const params = gpuContext.createBindGroup(optLayout, { a, b, c, cfg });
  optMatmulRunner
    .with(params)
    .dispatchWorkgroups(Math.ceil(n / OPT_BN), Math.ceil(m / OPT_BM));
  await flush();
  const got = (await c.read()).values as number[];
  const want = cpuRef(A, B, m, k, n);
  let maxErr = 0;
  for (let i = 0; i < m * n; i++)
    maxErr = Math.max(maxErr, Math.abs(got[i]! - want[i]!));
  return maxErr;
};

const measureBandwidth = async (elements: number) => {
  const x = storageBuffer(elements, true);
  const y = storageBuffer(elements, true);
  const out = storageBuffer(elements, false);
  const cfg = gpuContext.createBuffer(bwCfg, { n: elements }).$usage("uniform");
  const params = gpuContext.createBindGroup(bwLayout, { x, y, out, cfg });
  const groups = Math.min(Math.ceil(elements / 256), 32768);
  const dispatch = () => saxpyRunner.with(params).dispatchWorkgroups(groups);
  const ms = await timeDispatch(dispatch);
  return (elements * 3 * 4) / (ms / 1000) / 1e9;
};

const row = (label: string, cur: number, opt: number) =>
  `    ${label.padEnd(22)}${String(Math.round(cur)).padStart(8)}${String(
    Math.round(opt),
  ).padStart(8)}     ${((cur / opt) * 100).toFixed(1).padStart(6)}%`;

const main = async () => {
  if (maybeDumpWgsl({ dotProductKernel, saxpyKernel, optMatmulKernel })) {
    process.exit(0);
  }

  console.log(
    "matmul roofline bench — current vs edge-safe tuned-SGEMM ceiling\n",
  );

  // sanity: the ceiling kernel must be correct on ragged (non-tile-aligned) sizes
  const ragged = [
    [100, 77, 53],
    [200, 256, 13],
    [1000, 1000, 1000],
  ];
  console.log("  edge-safe correctness (max abs err vs CPU):");
  for (const [m, k, n] of ragged) {
    const err = await verifyOpt(m!, k!, n!);
    console.log(`    ${`${m}x${k} @ ${k}x${n}`.padEnd(22)} maxErr=${err.toExponential(2)}`);
  }

  const gbs = await measureBandwidth(16 * 1024 * 1024);

  // ideal-conditions ceiling: large, perfectly tile-aligned square matmuls
  let optBest = 0;
  console.log("\n  square sweep (M=K=N, tile-aligned)   current  opt-ceil   cur%");
  for (const nn of [256, 512, 1024, 2048]) {
    const cur = await measureCurrent(nn, nn, nn);
    const opt = await measureOpt(nn, nn, nn);
    optBest = Math.max(optBest, opt);
    console.log(row(`${nn}x${nn}x${nn}`, cur, opt));
  }

  // realistic: the model's actual layer shapes (hidden=256, mlp=1024, vocab=13)
  // swept over a realistic batched row count M (batch * tokens).
  console.log(
    "\n  real model-layer shapes (hidden=256 mlp=1024 vocab=13), M = batch*tokens",
  );
  console.log("    layer (MxKxN)          current  opt-ceil   cur%");
  const layers: [string, number, number][] = [
    ["proj K256 N256", 256, 256],
    ["mlp-up K256 N1024", 256, 1024],
    ["mlp-down K1024 N256", 1024, 256],
    ["logits K256 N13", 256, 13],
  ];
  for (const m of [512, 4096]) {
    console.log(`    --- M=${m} ---`);
    for (const [name, k, n] of layers) {
      const cur = await measureCurrent(m, k, n);
      const opt = await measureOpt(m, k, n);
      console.log(row(name, cur, opt));
    }
  }
  // ragged batch (M not a multiple of the 64-row tile) — the common real case
  console.log("    --- M=1000 (ragged) ---");
  for (const [name, k, n] of layers) {
    const cur = await measureCurrent(1000, k, n);
    const opt = await measureOpt(1000, k, n);
    console.log(row(name, cur, opt));
  }

  console.log("");
  console.log(
    `  ideal ceiling (large aligned square): ${optBest.toFixed(0)} GFLOP/s`,
  );
  console.log(
    `  streaming bandwidth: ${gbs.toFixed(0)} GB/s -> naive 0.25 FLOP/byte roofline ~= ${(gbs * 0.25).toFixed(0)} GFLOP/s`,
  );

  process.exit(0);
};

main();
