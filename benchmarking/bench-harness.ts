export const WARMUP_ITERS = 3;
export const MEASURE_ITERS = 10;

export type Stats = {
  mean: number;
  median: number;
  min: number;
  max: number;
  stddev: number;
};

export const computeStats = (samples: number[]): Stats => {
  const sorted = [...samples].sort((a, b) => a - b);
  const mean = samples.reduce((s, v) => s + v, 0) / samples.length;
  const median = sorted[Math.floor(sorted.length / 2)]!;
  const min = sorted[0]!;
  const max = sorted[sorted.length - 1]!;
  const variance =
    samples.reduce((s, v) => s + (v - mean) ** 2, 0) / samples.length;
  const stddev = Math.sqrt(variance);
  return { mean, median, min, max, stddev };
};

export const fmtMs = (ms: number) => `${ms.toFixed(3)}ms`;

export const benchmark = async (
  fn: () => void | Promise<void>,
): Promise<Stats> => {
  for (let i = 0; i < WARMUP_ITERS; i++) await fn();

  const samples: number[] = [];
  for (let i = 0; i < MEASURE_ITERS; i++) {
    const start = performance.now();
    await fn();
    samples.push(performance.now() - start);
  }
  return computeStats(samples);
};

export const printRow = (
  label: string,
  baselineLabel: string,
  baseline: Stats,
  candidateLabel: string,
  candidate: Stats,
  speedup: number,
) => {
  const speedupStr =
    speedup >= 1
      ? `${speedup.toFixed(2)}x faster`
      : `${(1 / speedup).toFixed(2)}x slower`;
  console.log(`\n  ${label}`);
  console.log(
    `    ${baselineLabel.padEnd(10)} mean=${fmtMs(baseline.mean)} median=${fmtMs(baseline.median)} min=${fmtMs(baseline.min)} stddev=${fmtMs(baseline.stddev)}`,
  );
  console.log(
    `    ${candidateLabel.padEnd(10)} mean=${fmtMs(candidate.mean)} median=${fmtMs(candidate.median)} min=${fmtMs(candidate.min)} stddev=${fmtMs(candidate.stddev)}`,
  );
  console.log(`    -> ${candidateLabel} is ${speedupStr} (median)`);
};

export const matricesMatch = (
  m1: number[][],
  m2: number[][],
  tolerance: number,
): { ok: true } | { ok: false; reason: string } => {
  if (m1.length !== m2.length)
    return { ok: false, reason: `row count: ${m1.length} vs ${m2.length}` };
  for (let i = 0; i < m1.length; i++) {
    const r1 = m1[i]!;
    const r2 = m2[i]!;
    if (r1.length !== r2.length)
      return {
        ok: false,
        reason: `row ${i} length: ${r1.length} vs ${r2.length}`,
      };
    for (let j = 0; j < r1.length; j++) {
      const diff = Math.abs(r1[j]! - r2[j]!);
      if (diff > tolerance) {
        return {
          ok: false,
          reason: `cell [${i},${j}]: ${r1[j]} vs ${r2[j]} (diff ${diff})`,
        };
      }
    }
  }
  return { ok: true };
};
