import { describe, expect, it } from "vitest";
import { softmax } from "../../shared/math.ts";
import { softmaxBackprop } from "./softmaxBackprop.ts";

describe("softmaxBackprop", () => {
  const softmaxObjective = (inputs: number[], outputGradients: number[]) => {
    const probabilities = softmax(inputs);

    return probabilities.reduce(
      (total, probability, index) =>
        total + probability * outputGradients[index]!,
      0,
    );
  };

  it("accounts for every softmax output depending on every input", () => {
    const inputs = [0, Math.log(2)];
    const outputGradients = [3, 5];

    const gradients = softmaxBackprop(inputs, outputGradients);

    expect(gradients[0]).toBeCloseTo(-4 / 9, 10);
    expect(gradients[1]).toBeCloseTo(4 / 9, 10);
  });

  it("matches finite differences for plain softmax backprop", () => {
    const inputs = [1.2, -0.4, 0.7];
    const outputGradients = [0.5, -1.3, 2.1];
    const epsilon = 0.000001;

    const gradients = softmaxBackprop(inputs, outputGradients);

    for (const [inputIndex] of inputs.entries()) {
      const increasedInputs = [...inputs];
      const decreasedInputs = [...inputs];

      increasedInputs[inputIndex]! += epsilon;
      decreasedInputs[inputIndex]! -= epsilon;

      const numericalGradient =
        (softmaxObjective(increasedInputs, outputGradients) -
          softmaxObjective(decreasedInputs, outputGradients)) /
        (2 * epsilon);

      expect(gradients[inputIndex]).toBeCloseTo(numericalGradient, 5);
    }
  });
});
