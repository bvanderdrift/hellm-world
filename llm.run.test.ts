import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.resetModules();
  vi.restoreAllMocks();
  vi.doUnmock("./matrices.ts");
  vi.doUnmock("./mlp.ts");
});

describe("runLlm wiring", () => {
  it("adds the MLP update back into the residual stream before unembedding", async () => {
    const mlpUpdateMatrix = [[10], [98]];
    const transformedState = [[11], [99]];
    const getMultilayerPerceptronUpdateMatrix = vi.fn(() => mlpUpdateMatrix);
    const addMatrices = vi.fn(() => transformedState);
    const multiplyMatrices = vi.fn(() => [
      [100, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 100, 0],
    ]);

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      addMatrices,
      multiplyMatrices,
    }));
    vi.doMock("./mlp.ts", () => ({
      getMultilayerPerceptronUpdateMatrix,
    }));

    const { runLlm } = await import("./llm.ts");
    const predictedToken = runLlm("hello world");

    expect(getMultilayerPerceptronUpdateMatrix).toHaveBeenCalledWith(
      [[1], [1]],
      expect.objectContaining({
        wUp: expect.any(Object),
        wDown: expect.any(Object),
      }),
    );
    expect(addMatrices).toHaveBeenCalledWith([[1], [1]], mlpUpdateMatrix);
    expect(multiplyMatrices).toHaveBeenCalledWith(
      transformedState,
      expect.any(Array),
    );
    expect(predictedToken).toBe("is");
  });

  it("maps the winning logit index to the vocabulary, not the prompt position", async () => {
    const getMultilayerPerceptronUpdateMatrix = vi.fn(() => [[0], [0]]);
    const addMatrices = vi.fn((state: number[][]) => state);

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      addMatrices,
      multiplyMatrices: () => {
        return [
          [0, 0, 0, 0, 100, 0],
          [0, 0, 0, 0, 100, 0],
        ];
      },
    }));
    vi.doMock("./mlp.ts", () => ({
      getMultilayerPerceptronUpdateMatrix,
    }));

    const { runLlm } = await import("./llm.ts");

    expect(runLlm("hello world")).toBe("is");
    expect(addMatrices).toHaveBeenCalledWith([[1], [1]], [[0], [0]]);
  });
});
