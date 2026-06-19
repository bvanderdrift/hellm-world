/** THIS FILE IS AI GENERATED */
import tgpu, { d } from "typegpu";
import { builtin } from "typegpu/data";
import * as std from "typegpu/std";
import { gpuContext } from "../shared/gpu-context.ts";
import { createMatrix } from "../shared/matrices.ts";
import {
  createMatrixBuffer,
  multiplyMatricesOnGPU,
} from "../shared/matrices-gpu.ts";

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

  const b = d.vec4f(0.9999999, 0.9999999, 0.9999999, 0.9999999);
  const c = d.vec4f(0.0000001, 0.0000001, 0.0000001, 0.0000001);

  let a0 = d.vec4f(0.1, 0.2, 0.3, 0.4);
  let a1 = d.vec4f(0.2, 0.3, 0.4, 0.5);
  let a2 = d.vec4f(0.3, 0.4, 0.5, 0.6);
  let a3 = d.vec4f(0.4, 0.5, 0.6, 0.7);
  let a4 = d.vec4f(0.5, 0.6, 0.7, 0.8);
  let a5 = d.vec4f(0.6, 0.7, 0.8, 0.9);
  let a6 = d.vec4f(0.7, 0.8, 0.9, 1.0);
  let a7 = d.vec4f(0.8, 0.9, 1.0, 1.1);

  for (let i = d.u32(0); i < cfg.iters; i++) {
    a0 = std.add(std.mul(a0, b), c);
    a1 = std.add(std.mul(a1, b), c);
    a2 = std.add(std.mul(a2, b), c);
    a3 = std.add(std.mul(a3, b), c);
    a4 = std.add(std.mul(a4, b), c);
    a5 = std.add(std.mul(a5, b), c);
    a6 = std.add(std.mul(a6, b), c);
    a7 = std.add(std.mul(a7, b), c);
  }

  const sum = std.add(
    std.add(std.add(a0, a1), std.add(a2, a3)),
    std.add(std.add(a4, a5), std.add(a6, a7)),
  );
  peakLayout.$.out.values[idx]! = sum.x + sum.y + sum.z + sum.w;
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

const measureMatmul = async (m: number, k: number, n: number) => {
  const a = createMatrixBuffer(createMatrix(m, k, rand));
  const b = createMatrixBuffer(createMatrix(k, n, rand));
  const out = createMatrixBuffer(createMatrix(m, n));

  const dispatch = () => multiplyMatricesOnGPU(a, b, out);

  const ms = await timeDispatch(dispatch);
  const flops = 2 * m * n * k;
  return { ms, gflops: flops / (ms / 1000) / 1e9 };
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
  const bytes = elements * 3 * 4;
  return bytes / (ms / 1000) / 1e9;
};

const measurePeak = async (threads: number, innerIters: number) => {
  const out = storageBuffer(threads, false);
  const cfg = gpuContext
    .createBuffer(peakCfg, { n: threads, iters: innerIters })
    .$usage("uniform");
  const params = gpuContext.createBindGroup(peakLayout, { out, cfg });
  const groups = Math.min(Math.ceil(threads / 256), 32768);
  const dispatch = () => peakRunner.with(params).dispatchWorkgroups(groups);

  const ms = await timeDispatch(dispatch);
  const flops = threads * innerIters * 8 * 4 * 2;
  return flops / (ms / 1000) / 1e9;
};

const main = async () => {
  console.log("matmul roofline bench (current tiled multiplyMatricesOnGPU)\n");

  const gbs = await measureBandwidth(16 * 1024 * 1024);
  const peak = await measurePeak(1 << 20, 4096);

  const sizes = [128, 256, 512, 1024, 2048];

  console.log("  square matmul sweep (M=K=N)");
  console.log(
    "       N    ms/dispatch     GFLOP/s    % of peak    % of BW-roofline",
  );

  let best = 0;
  for (const n of sizes) {
    const { ms, gflops } = await measureMatmul(n, n, n);
    best = Math.max(best, gflops);
    const pctPeak = (gflops / peak) * 100;
    const pctRoofline = (gflops / (gbs * 0.25)) * 100;
    console.log(
      `    ${String(n).padStart(4)}    ${ms.toFixed(3).padStart(9)}ms    ${gflops.toFixed(0).padStart(7)}     ${pctPeak.toFixed(1).padStart(6)}%     ${pctRoofline.toFixed(0).padStart(6)}%`,
    );
  }

  console.log("");
  console.log(`  best matmul: ${best.toFixed(0)} GFLOP/s FP32`);
  console.log(
    `  compute-peak probe: ${peak.toFixed(0)} GFLOP/s (matmul reaches ${((best / peak) * 100).toFixed(1)}%)`,
  );
  console.log(
    `  streaming bandwidth: ${gbs.toFixed(0)} GB/s -> naive 0.25 FLOP/byte roofline ~= ${(gbs * 0.25).toFixed(0)} GFLOP/s`,
  );

  process.exit(0);
};

main();
