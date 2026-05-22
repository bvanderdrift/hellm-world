import { describe, expect, it } from "vitest";
import { softmax } from "../../shared/math.ts";
import { softmaxBackprop } from "./softmaxBackprop.ts";
import {
  FINITE_DIFFERENCE_EPSILON,
  FINITE_DIFFERENCE_PRECISION,
  TESTING_PRECISION,
} from "../../testing/constants.ts";

describe("softmaxBackprop", () => {
  const softmaxObjective = (
    inputs: Float32Array,
    outputGradients: Float32Array,
  ) => {
    const probabilities = softmax(inputs);

    return probabilities.reduce(
      (total, probability, index) =>
        total + probability * outputGradients[index]!,
      0,
    );
  };

  it("accounts for every softmax output depending on every input", () => {
    const inputs = new Float32Array([0, Math.log(2)]);
    const outputGradients = new Float32Array([3, 5]);

    const gradients = softmaxBackprop(inputs, outputGradients);

    expect(gradients[0]).toBeCloseTo(-4 / 9, TESTING_PRECISION);
    expect(gradients[1]).toBeCloseTo(4 / 9, TESTING_PRECISION);
  });

  it("matches finite differences for plain softmax backprop", () => {
    const inputs = new Float32Array([1.2, -0.4, 0.7]);
    const outputGradients = new Float32Array([0.5, -1.3, 2.1]);

    const gradients = softmaxBackprop(inputs, outputGradients);

    for (const [inputIndex] of inputs.entries()) {
      const increasedInputs = new Float32Array([...inputs]);
      const decreasedInputs = new Float32Array([...inputs]);

      increasedInputs[inputIndex]! += FINITE_DIFFERENCE_EPSILON;
      decreasedInputs[inputIndex]! -= FINITE_DIFFERENCE_EPSILON;

      const numericalGradient =
        (softmaxObjective(increasedInputs, outputGradients) -
          softmaxObjective(decreasedInputs, outputGradients)) /
        (2 * FINITE_DIFFERENCE_EPSILON);

      expect(gradients[inputIndex]).toBeCloseTo(
        numericalGradient,
        FINITE_DIFFERENCE_PRECISION,
      );
    }
  });
});
