import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.resetModules();
  vi.restoreAllMocks();
  vi.doUnmock("./matrices.ts");
});

describe("runLlm wiring", () => {
  it("uses the logits from the last context position for next-token prediction", async () => {
    let multiplyCallCount = 0;

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      multiplyMatrices: () => {
        multiplyCallCount += 1;

        if (multiplyCallCount === 1) {
          return [[1], [1]];
        }

        return [
          [100, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 100],
        ];
      },
    }));

    const { runLlm } = await import("./llm.ts");

    expect(runLlm("hello world")).toBe("beer");
  });

  it("maps the winning logit index to the vocabulary, not the prompt position", async () => {
    let multiplyCallCount = 0;

    vi.doMock("./matrices.ts", () => ({
      validateSize: () => {},
      multiplyMatrices: () => {
        multiplyCallCount += 1;

        if (multiplyCallCount === 1) {
          return [[1], [1]];
        }

        return [
          [0, 0, 0, 0, 100, 0],
          [0, 0, 0, 0, 100, 0],
        ];
      },
    }));

    const { runLlm } = await import("./llm.ts");

    expect(runLlm("hello world")).toBe("is");
  });
});
