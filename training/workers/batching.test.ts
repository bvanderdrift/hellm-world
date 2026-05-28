import { describe, expect, it } from "vitest";
import { splitAcrossWorkers } from "./batching.ts";
import type { TrainingExample } from "../doSingleTrainingPass.ts";

const makeExamples = (count: number): TrainingExample[] =>
  Array.from({ length: count }, (_, i) => ({
    sequence: [`token_${i}`],
    maskBeforeIndex: null,
  }));

describe("splitAcrossWorkers", () => {
  it("splits evenly when data divides perfectly", () => {
    const result = splitAcrossWorkers(makeExamples(9), 3);

    expect(result).toHaveLength(3);
    expect(result[0]).toHaveLength(3);
    expect(result[1]).toHaveLength(3);
    expect(result[2]).toHaveLength(3);
  });

  it("distributes remainder across first workers", () => {
    const result = splitAcrossWorkers(makeExamples(10), 3);

    expect(result).toHaveLength(3);
    expect(result[0]).toHaveLength(4);
    expect(result[1]).toHaveLength(3);
    expect(result[2]).toHaveLength(3);
  });

  it("never produces empty batches when data >= workers", () => {
    const result = splitAcrossWorkers(makeExamples(100), 16);

    expect(result).toHaveLength(16);
    for (const batch of result) {
      expect(batch.length).toBeGreaterThan(0);
    }
  });

  it("preserves all items with no duplicates", () => {
    const examples = makeExamples(100);
    const result = splitAcrossWorkers(examples, 16);

    const flattened = result.flat();
    expect(flattened).toHaveLength(100);
    expect(flattened).toEqual(examples);
  });

  it("handles more workers than data", () => {
    const result = splitAcrossWorkers(makeExamples(3), 16);

    expect(result).toHaveLength(16);
    const nonEmpty = result.filter((b) => b.length > 0);
    expect(nonEmpty).toHaveLength(3);
    expect(result.flat()).toHaveLength(3);
  });

  it("handles single worker", () => {
    const examples = makeExamples(10);
    const result = splitAcrossWorkers(examples, 1);

    expect(result).toHaveLength(1);
    expect(result[0]).toEqual(examples);
  });
});
