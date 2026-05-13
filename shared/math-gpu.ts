import { gpuContext } from "./gpu-context.ts";
import { singleMatrixParamsLayout, type MatrixBuffer } from "./matrices-gpu.ts";

export const reluOnGpu = (matrix: MatrixBuffer) => {
  const params = gpuContext.createBindGroup(singleMatrixParamsLayout, {
    m: matrix.buffer,
  });

  gpuContext
    .createGuardedComputePipeline((i: number) => {
      "use gpu";
      const m = singleMatrixParamsLayout.$.m;

      const currentValue = m.values[i]!;

      if (currentValue > 0) {
        m.values[i]! = currentValue;
      } else {
        m.values[i]! = 0;
      }
    })
    .with(params)
    .dispatchThreads(matrix.vectors * matrix.dimensions);
};

const sumOnGPU = (values: number[]) => {
  // "use gpu";

  let sum = 0;

  for (let index = 0; index < values.length; index++) {
    sum += values[index]!;
  }

  return sum;
};

const meanOnGPU = (values: number[]) => {
  // "use gpu";
  return sumOnGPU(values) / values.length;
};

const safeSumExponatedLogitsOnGPU = (logits: number[]) => {
  // "use gpu";
  const biggestLogit = Math.max(...logits);
  // To prevent overflows. Logits are still related the same since they all subtract the same value
  const safeLogits = logits.map((logit) => logit - biggestLogit);
  const exponatedLogits = safeLogits.map((l) => Math.exp(l));

  return {
    safeLogits,
    exponatedLogits,
    summed: sumOnGPU(exponatedLogits),
    biggestLogit,
  };
};

/**
 * s_i = e^l_i / sum(e^l_j)
 */
export const softmaxOnGpu = (logits: number[]) => {
  // "use gpu";
  const { safeLogits, summed } = safeSumExponatedLogitsOnGPU(logits);

  return safeLogits.map((logit) => Math.exp(logit) / summed);
};
