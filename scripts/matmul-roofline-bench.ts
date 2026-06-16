/** THIS FILE IS AI GENERATED */
import tgpu, { d } from "typegpu";
import { builtin } from "typegpu/data";
import { gpuContext } from "../shared/gpu-context.ts";

const rand = () => Math.random() * 2 - 1;

const matCfg = d.struct({
  m: d.u32,
  k: d.u32,
  n: d.u32,
  r: d.u32,
});

const bwCfg = d.struct({
  n: d.u32,
});

const arr = (length: number) => d.struct({ values: d.arrayOf(d.f32, length) });
const arrSlot = arr(0);

const matLayout = tgpu.bindGroupLayout({
  m1: { storage: arrSlot, access: "readonly" },
  m2: { storage: arrSlot, access: "readonly" },
  mOut: { storage: arrSlot, access: "mutable" },
  cfg: { uniform: matCfg },
});

const matmulKernel = tgpu.computeFn({
  in: { globalId: builtin.globalInvocationId },
  workgroupSize: [16, 16],
})((input) => {
  "use gpu";

  const cfg = matLayout.$.cfg;
  const i = input.globalId.x;
  const j = input.globalId.y;

  if (i >= cfg.m || j >= cfg.n) {
    return;
  }

  const m1 = matLayout.$.m1;
  const m2 = matLayout.$.m2;
  const mOut = matLayout.$.mOut;

  const decay = d.f32(0.9999999);
  let summed = d.f32(0);

  for (let k = d.u32(0); k < cfg.k; k++) {
    const v1 = m1.values[i * cfg.k + k]!;
    const v2 = m2.values[k * cfg.n + j]!;
    for (let r = d.u32(0); r < cfg.r; r++) {
      summed = summed * decay + v1 * v2;
    }
  }

  mOut.values[i * cfg.n + j]! = summed;
});

const matmulRunner = gpuContext.createComputePipeline({
  compute: matmulKernel,
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

const peakCfg = d.struct({
  n: d.u32,
  iters: d.u32,
});

const peakLayout = tgpu.bindGroupLayout({
  out: { storage: arrSlot, access: "mutable" },
  cfg: { uniform: peakCfg },
});

const peakKernel = tgpu.computeFn({
  in: { globalId: builtin.globalInvocationId },
  workgroupSize: [256],
})((input) => {
  "use gpu";

  const cfg = peakLayout.$.cfg;
  const idx = input.globalId.x;

  if (idx >= cfg.n) {
    return;
  }

  const b = d.f32(0.9999999);
  const c = d.f32(0.0000001);

  let a0 = d.f32(0.1);
  let a1 = d.f32(0.2);
  let a2 = d.f32(0.3);
  let a3 = d.f32(0.4);
  let a4 = d.f32(0.5);
  let a5 = d.f32(0.6);
  let a6 = d.f32(0.7);
  let a7 = d.f32(0.8);

  for (let i = d.u32(0); i < cfg.iters; i++) {
    a0 = a0 * b + c;
    a1 = a1 * b + c;
    a2 = a2 * b + c;
    a3 = a3 * b + c;
    a4 = a4 * b + c;
    a5 = a5 * b + c;
    a6 = a6 * b + c;
    a7 = a7 * b + c;
  }

  peakLayout.$.out.values[idx]! = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
});

const peakRunner = gpuContext.createComputePipeline({ compute: peakKernel });

const flush = () => gpuContext.device.queue.onSubmittedWorkDone();

const storageBuffer = (length: number, fill: boolean) => {
  const builder = gpuContext.createBuffer(
    arr(length),
    fill ? { values: Array.from({ length }, rand) } : undefined,
  );
  return builder.$usage("storage");
};

const timeDispatch = async (
  dispatch: () => void,
  warmup: number,
  iters: number,
) => {
  for (let i = 0; i < warmup; i++) dispatch();
  await flush();

  const start = performance.now();
  for (let i = 0; i < iters; i++) dispatch();
  await flush();
  return (performance.now() - start) / iters;
};

const parseArgs = () => {
  const args = process.argv.slice(2);
  const get = (flag: string, fallback: number) => {
    const idx = args.indexOf(flag);
    if (idx === -1) return fallback;
    return Number(args[idx + 1]);
  };
  return {
    m: get("--m", 8192),
    k: get("--k", 512),
    n: get("--n", 512),
    iters: get("--iters", 20),
    warmup: get("--warmup", 5),
    bwElements: get("--bw", 16 * 1024 * 1024),
    bwIters: get("--bw-iters", 30),
    peakThreads: get("--peak-threads", 1 << 20),
    peakIters: get("--peak-iters", 4096),
  };
};

const main = async () => {
  const {
    m,
    k,
    n,
    iters,
    warmup,
    bwElements,
    bwIters,
    peakThreads,
    peakIters,
  } = parseArgs();

  console.log("matmul roofline bench");
  console.log(
    `  matmul M=${m} K=${k} N=${n} · warmup=${warmup} measure=${iters}`,
  );
  console.log(
    `  bandwidth elements=${bwElements.toLocaleString()} · measure=${bwIters}\n`,
  );

  const m1 = storageBuffer(m * k, true);
  const m2 = storageBuffer(k * n, true);
  const mOut = storageBuffer(m * n, false);

  const groupsX = Math.ceil(m / 16);
  const groupsY = Math.ceil(n / 16);

  const redundancies = [1, 2, 4, 8, 16, 32];
  const results: { r: number; ms: number; gflops: number }[] = [];

  console.log("  redundancy sweep (same memory traffic, R x the FLOPs)");
  console.log(
    "    R     ms/dispatch    GFLOP/s     vs R=1    (flat => memory-bound)",
  );

  for (const r of redundancies) {
    const cfg = gpuContext
      .createBuffer(matCfg, { m, k, n, r })
      .$usage("uniform");

    const params = gpuContext.createBindGroup(matLayout, {
      m1,
      m2,
      mOut,
      cfg,
    });

    const dispatch = () =>
      matmulRunner.with(params).dispatchWorkgroups(groupsX, groupsY);

    const ms = await timeDispatch(dispatch, warmup, iters);
    const flops = 2 * m * n * k * r;
    const gflops = flops / (ms / 1000) / 1e9;
    results.push({ r, ms, gflops });

    const ratio = ms / results[0]!.ms;
    console.log(
      `    ${String(r).padStart(2)}    ${ms.toFixed(3).padStart(9)}ms    ${gflops.toFixed(0).padStart(7)}     ${ratio.toFixed(2).padStart(5)}x`,
    );
  }

  const baseGflops = results[0]!.gflops;
  const last = results[results.length - 1]!;

  const xs = results.map((x) => x.r);
  const ys = results.map((x) => x.ms);
  const count = xs.length;
  const sumX = xs.reduce((a, b) => a + b, 0);
  const sumY = ys.reduce((a, b) => a + b, 0);
  const sumXX = xs.reduce((a, b) => a + b * b, 0);
  const sumXY = xs.reduce((a, b, i) => a + b * ys[i]!, 0);
  const slope = (count * sumXY - sumX * sumY) / (count * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / count;
  const fixedFraction = intercept / results[0]!.ms;

  const flopsPerR = 2 * m * n * k;
  const serialAsymptote = flopsPerR / (slope / 1000) / 1e9;

  console.log("");
  console.log(`  R=1 (real naive matmul): ${baseGflops.toFixed(0)} GFLOP/s`);
  console.log(
    `  last sample R=${last.r}: ${last.gflops.toFixed(0)} GFLOP/s (still rising -> NOT a ceiling)`,
  );
  console.log(
    `  linear fit: time(R) ~= ${intercept.toFixed(1)}ms fixed + ${slope.toFixed(1)}ms x R`,
  );
  console.log(
    `  serial-chain asymptote (R->inf) = flops_per_R / slope = ${serialAsymptote.toFixed(0)} GFLOP/s`,
  );
  console.log(
    `    (LATENCY-bound ceiling of a 1-accumulator chain, NOT the true ALU throughput)`,
  );
  console.log(
    `  at R=1: ~${(fixedFraction * 100).toFixed(0)}% fixed cost (memory loads / loop / latency, no extra arithmetic)`,
  );
  console.log(
    `          ~${((1 - fixedFraction) * 100).toFixed(0)}% arithmetic (latency-inflated upper bound on true compute)`,
  );

  const bx = storageBuffer(bwElements, true);
  const by = storageBuffer(bwElements, true);
  const bout = storageBuffer(bwElements, false);
  const bcfg = gpuContext
    .createBuffer(bwCfg, { n: bwElements })
    .$usage("uniform");
  const bparams = gpuContext.createBindGroup(bwLayout, {
    x: bx,
    y: by,
    out: bout,
    cfg: bcfg,
  });
  const bwGroups = Math.min(Math.ceil(bwElements / 256), 32768);
  const bwDispatch = () =>
    saxpyRunner.with(bparams).dispatchWorkgroups(bwGroups);

  const bwMs = await timeDispatch(bwDispatch, warmup, bwIters);
  const bwBytes = bwElements * 3 * 4;
  const gbs = bwBytes / (bwMs / 1000) / 1e9;

  console.log("");
  console.log("  bandwidth anchor (saxpy: 2 reads + 1 write per element)");
  console.log(
    `    ${bwMs.toFixed(3)}ms/dispatch -> ${gbs.toFixed(0)} GB/s measured streaming bandwidth`,
  );
  console.log(
    `    naive-matmul roofline at 0.25 FLOP/byte ~= ${(gbs * 0.25).toFixed(0)} GFLOP/s`,
  );
  console.log(
    `    (if R=1 GFLOP/s sits near this line, the naive kernel is pinned to memory bandwidth)\n`,
  );

  const pOut = storageBuffer(peakThreads, false);
  const pCfg = gpuContext
    .createBuffer(peakCfg, { n: peakThreads, iters: peakIters })
    .$usage("uniform");
  const pParams = gpuContext.createBindGroup(peakLayout, {
    out: pOut,
    cfg: pCfg,
  });
  const pGroups = Math.min(Math.ceil(peakThreads / 256), 32768);
  const pDispatch = () => peakRunner.with(pParams).dispatchWorkgroups(pGroups);

  const pMs = await timeDispatch(pDispatch, warmup, iters);
  const pFlops = peakThreads * peakIters * 8 * 2;
  const pGflops = pFlops / (pMs / 1000) / 1e9;

  console.log(
    "  compute-throughput probe (8 independent accumulators -> true ALU ceiling)",
  );
  console.log(
    `    threads=${peakThreads.toLocaleString()} iters=${peakIters} -> ${pGflops.toFixed(0)} GFLOP/s sustained FP32`,
  );
  console.log(
    `    real kernel R=1 (${baseGflops.toFixed(0)} GFLOP/s) is ${((baseGflops / pGflops) * 100).toFixed(1)}% of this -> ~${(pGflops / baseGflops).toFixed(0)}x idle ALU headroom\n`,
  );

  process.exit(0);
};

main();
