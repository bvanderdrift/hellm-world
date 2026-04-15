import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.resetModules();
  vi.restoreAllMocks();
  vi.doUnmock("./matrices.ts");
  vi.doUnmock("./transform.ts");
});

describe("runLlm wiring", () => {
  it("unembeds the transformed hidden state and uses the last context position for next-token prediction", async () => {
    const transformedState = [[11], [99]];
    const runMultilayerPerceptronOnMatrix = vi.fn(() => transformedState);

    const multiplyMatrices = vi.fn((state: number[][]) => {
      if (state === transformedState) {
        return [
          [100, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 100, 0],
        ];
      }

      return [
        [0, 0, 0, 0, 0, 100],
        [100, 0, 0, 0, 0, 0],
      ];
    });

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      multiplyMatrices,
    }));
    vi.doMock("./transform.ts", () => ({
      runMultilayerPerceptronOnMatrix,
    }));

    const { runLlm } = await import("./llm.ts");
    const predictedToken = runLlm("hello world");

    expect(runMultilayerPerceptronOnMatrix).toHaveBeenCalledWith(
      [[1], [1]],
      expect.objectContaining({
        wUp: expect.any(Object),
        wDown: expect.any(Object),
      }),
    );
    expect(multiplyMatrices).toHaveBeenCalledWith(
      transformedState,
      expect.any(Array),
    );
    expect(predictedToken).toBe("is");
  });

  it("maps the winning logit index to the vocabulary, not the prompt position", async () => {
    const runMultilayerPerceptronOnMatrix = vi.fn((state: number[][]) => state);

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      multiplyMatrices: () => {
        return [
          [0, 0, 0, 0, 100, 0],
          [0, 0, 0, 0, 100, 0],
        ];
      },
    }));
    vi.doMock("./transform.ts", () => ({
      runMultilayerPerceptronOnMatrix,
    }));

    const { runLlm } = await import("./llm.ts");

    expect(runLlm("hello world")).toBe("is");
  });
});
