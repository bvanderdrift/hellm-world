import { safeSumExponatedLogits, sum } from "../../shared/math.ts";

/**
 * s_i = e^x_i / sum(e^x_j)
 *   = 1/sum(e^x_j) * e^x_i
 *   = sum(e^x_j)^-1 * e^x_i
 *
 * dL/dx_i = sum(dL/ds_j * ds_j/dx_i) because x_i influences ALL output
 *
 * ds_j/dx_i =
 *  if i === j : ds_i/dx_i = d(sum(e^x_k)^-1)/dx_i * e^x_i + 1/sum(e^x_k) * d(e^x_i)/dx_i
 *     = -sum(e^x_k)^-2 * e^x_i * e^x_i + 1/sum(e^x_k) * e^x_i
 *     = -(e^x_i)^2 / sum(e^x_k)^2 + e^x_i/sum(e^x_k)
 *
 *  if i !== j : ds_j/dx_i = d(sum(e^x_k)^-1)/dx_i * e^x_j + 1/sum(e^x_k) * d(e^x_j)/dx_i
 *     = -sum(e^x_k)^-2 * e^x_i * e^x_j + 1/sum(e^x_k) * 0
 *     = - (e^x_i * e^x_j) / sum(e^x_k)^2
 *
 * because
 * d(sum(e^x_j)^-1)/dx_i = d(sum(e^x_j)^-1)/d(sum(e^x_j)) * d(sum(e^x_j))/dx_i
 *   = -sum(e^x_j)^-2 * e^x_i
 */
export const softmaxBackprop = (
  inputs: number[],
  outputGradients: number[],
) => {
  const { summed, exponatedLogits } = safeSumExponatedLogits(inputs);

  return exponatedLogits.map((epx_i, i) => {
    return sum(
      outputGradients.map((dLDs_j, j) => {
        const epx_j = exponatedLogits[j]!;

        const depx_jDepx_i = i === j ? epx_i : 0;

        const ds_jDx_i =
          -(epx_i / summed) * (epx_j / summed) + depx_jDepx_i / summed;

        return dLDs_j * ds_jDx_i;
      }),
    );
  });
};
