/** THIS FILE IS AI GENERATED */
import tgpu, { d } from "typegpu";
import { builtin } from "typegpu/data";
import * as std from "typegpu/std";
import { gpuContext } from "../shared/gpu-context.ts";

const arr = (length: number) => d.struct({ values: d.arrayOf(d.f32, length) });
const arrSlot = arr(0);

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
  let a8 = d.vec4f(0.9, 1.0, 1.1, 1.2);
  let a9 = d.vec4f(1.0, 1.1, 1.2, 1.3);
  let a10 = d.vec4f(1.1, 1.2, 1.3, 1.4);
  let a11 = d.vec4f(1.2, 1.3, 1.4, 1.5);
  let a12 = d.vec4f(1.3, 1.4, 1.5, 1.6);
  let a13 = d.vec4f(1.4, 1.5, 1.6, 1.7);
  let a14 = d.vec4f(1.5, 1.6, 1.7, 1.8);
  let a15 = d.vec4f(1.6, 1.7, 1.8, 1.9);
  for (let i = d.u32(0); i < cfg.iters; i++) {
    a0 = std.add(std.mul(a0, b), c);
    a1 = std.add(std.mul(a1, b), c);
    a2 = std.add(std.mul(a2, b), c);
    a3 = std.add(std.mul(a3, b), c);
    a4 = std.add(std.mul(a4, b), c);
    a5 = std.add(std.mul(a5, b), c);
    a6 = std.add(std.mul(a6, b), c);
    a7 = std.add(std.mul(a7, b), c);
    a8 = std.add(std.mul(a8, b), c);
    a9 = std.add(std.mul(a9, b), c);
    a10 = std.add(std.mul(a10, b), c);
    a11 = std.add(std.mul(a11, b), c);
    a12 = std.add(std.mul(a12, b), c);
    a13 = std.add(std.mul(a13, b), c);
    a14 = std.add(std.mul(a14, b), c);
    a15 = std.add(std.mul(a15, b), c);
  }

  const sum = std.add(
    std.add(
      std.add(std.add(a0, a1), std.add(a2, a3)),
      std.add(std.add(a4, a5), std.add(a6, a7)),
    ),
    std.add(
      std.add(std.add(a8, a9), std.add(a10, a11)),
      std.add(std.add(a12, a13), std.add(a14, a15)),
    ),
  );
  peakLayout.$.out.values[idx]! = sum.x + sum.y + sum.z + sum.w;
});

const peakRunner = gpuContext.createComputePipeline({ compute: peakKernel });

const flush = () => gpuContext.device.queue.onSubmittedWorkDone();

const storageBuffer = (length: number) => {
  const builder = gpuContext.createBuffer(arr(length));
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

const measure = async (threads: number, innerIters: number) => {
  const out = storageBuffer(threads);
  const cfg = gpuContext
    .createBuffer(peakCfg, { n: threads, iters: innerIters })
    .$usage("uniform");
  const params = gpuContext.createBindGroup(peakLayout, { out, cfg });
  const groups = Math.min(Math.ceil(threads / 256), 32768);
  const dispatch = () => peakRunner.with(params).dispatchWorkgroups(groups);

  const ms = await timeDispatch(dispatch);
  const flops = threads * innerIters * 16 * 4 * 2;
  return flops / (ms / 1000) / 1e9;
};

const main = async () => {
  const threadSweep = [
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
    1 << 23,
  ];
  const innerIters = 4096;

  let best = 0;
  for (const threads of threadSweep) {
    const gflops = await measure(threads, innerIters);
    best = Math.max(best, gflops);
    console.log(
      `threads=${threads.toLocaleString().padStart(11)}  ${gflops.toFixed(0).padStart(7)} GFLOP/s`,
    );
  }

  console.log(
    `\nmax: ${best.toFixed(0)} GFLOP/s (${(best / 1000).toFixed(2)} TFLOP/s) FP32`,
  );

  process.exit(0);
};

main();
