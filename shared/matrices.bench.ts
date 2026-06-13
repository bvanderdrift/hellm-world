import {
  createMatrixBufferAndCopy,
  multiplyMatricesOnGPU,
  type MatrixBuffer,
} from "./matrices-gpu.ts";
import {
  createMatrix,
  multiplyMatrices,
  type Matrix,
} from "./matrices.ts";
import { gpuContext } from "./gpu-context.ts";
import {
  benchmark,
  compareAcrossSizes,
  fmtMs,
  rand,
} from "../bench-harness.ts";

type Operands = { b: Matrix; bBuf: MatrixBuffer; out: MatrixBuffer };

const timeMatmul = async (m: number, k: number, n: number) => {
  const a = createMatrix(m, k, rand);
  const b = createMatrix(k, n, rand);
  const aBuf = createMatrixBufferAndCopy(a);
  const bBuf = createMatrixBufferAndCopy(b);
  const outBuf = createMatrixBufferAndCopy(createMatrix(m, n));

  const cpu = await benchmark(() => {
    multiplyMatrices(a, b);
  });
  const gpu = await benchmark(async () => {
    multiplyMatricesOnGPU(aBuf, bBuf, outBuf);
    await gpuContext.device.queue.onSubmittedWorkDone();
  });
  return { cpu: cpu.median, gpu: gpu.median };
};

const findCrossover = async (lo: number, hi: number): Promise<number> => {
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    const { cpu, gpu } = await timeMatmul(mid, mid, mid);
    console.log(
      `  probe n=${mid} (params=${2 * mid * mid}): cpu=${fmtMs(cpu)} gpu=${fmtMs(gpu)}`,
    );
    if (gpu < cpu) hi = mid;
    else lo = mid + 1;
  }
  return lo;
};

if (import.meta.main) {
  await compareAcrossSizes<Operands>({
    name: "matmul CPU vs GPU benchmark (A[v×d] * B[d×d])",
    tolerance: 1e-4,
    setup: ({ vectors, dimensions }) => {
      const b = createMatrix(dimensions, dimensions, rand);
      return {
        b,
        bBuf: createMatrixBufferAndCopy(b),
        out: createMatrixBufferAndCopy(createMatrix(vectors, dimensions)),
      };
    },
    cpu: ({ matrix }, { b }) => multiplyMatrices(matrix, b),
    gpu: ({ buffer }, { bBuf, out }) => {
      multiplyMatricesOnGPU(buffer, bBuf, out);
      return out;
    },
  });

  console.log("--- crossover search (binary search on square dimension) ---");
  const crossoverN = await findCrossover(32, 512);
  console.log(
    `\n  crossover at n=${crossoverN} (square ${crossoverN}x${crossoverN}), parameter count=${2 * crossoverN * crossoverN}`,
  );

  console.log(
    "\n--- shape comparison (same ~param count, different shapes) ---",
  );
  const targetParams = 2 * crossoverN * crossoverN;
  const shapes: Array<{ label: string; m: number; k: number; n: number }> = [
    { label: "1×P * P×1 (dot product)", m: 1, k: targetParams / 2, n: 1 },
    { label: "4×K * K×4", m: 4, k: Math.round(targetParams / 8), n: 4 },
    { label: "16×K * K×16", m: 16, k: Math.round(targetParams / 32), n: 16 },
    { label: "64×K * K×64", m: 64, k: Math.round(targetParams / 128), n: 64 },
    { label: `${crossoverN}² (square)`, m: crossoverN, k: crossoverN, n: crossoverN },
    { label: "wide 1×K * K×256", m: 1, k: Math.round(targetParams / 257), n: 256 },
  ];

  console.log(`  target param count ≈ ${targetParams}\n`);
  for (const { label, m, k, n } of shapes) {
    const { cpu, gpu } = await timeMatmul(m, k, n);
    const ratio = cpu / gpu;
    console.log(
      `  ${label}  (${m}×${k} * ${k}×${n}, output=${m * n} cells)\n    cpu=${fmtMs(cpu)} gpu=${fmtMs(gpu)} (${ratio >= 1 ? ratio.toFixed(2) : (1 / ratio).toFixed(2)}x ${ratio >= 1 ? "GPU" : "CPU"} wins)`,
    );
  }
}
