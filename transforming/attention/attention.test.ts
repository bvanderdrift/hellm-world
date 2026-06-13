import { describe, it } from "vitest";
import {
  runSelfAttentionHead,
  runSelfAttentionMechanism,
} from "./attention.ts";
import type { AttentionWeights } from "../../model/model-types.ts";
import {
  matrixFrom,
  expectMatrixCloseTo,
} from "../../testing/testing-utils.ts";

const input = matrixFrom([
  [1, 0, 0, 0],
  [0, 1, 0, 0],
]);

describe("runSelfAttentionHead", () => {
  it("uses the value matrix as the payload that gets mixed across positions", () => {
    const output = runSelfAttentionHead(
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 10, 100, 1000],
        [2, 20, 200, 2000],
      ]),
      1,
      4,
    );

    expectMatrixCloseTo(
      output.output,
      matrixFrom([
        [1, 10, 100, 1000],
        [1.5, 15, 150, 1500],
      ]),
    );
  });

  it("uses query-key similarity to weight the visible values", () => {
    const output = runSelfAttentionHead(
      matrixFrom([[0], [Math.log(2)]]),
      matrixFrom([[1], [0]]),
      matrixFrom([[1], [2]]),
      1,
      1,
    );

    expectMatrixCloseTo(output.output, matrixFrom([[1], [4 / 3]]));
  });

  it("each head attends using its own column range of Q and K", () => {
    const output = runSelfAttentionHead(
      matrixFrom([
        [10, 0, 0, 0],
        [10, 0, 0, 0],
      ]),
      matrixFrom([
        [10, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      matrixFrom([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]),
      2,
      2,
    );

    expectMatrixCloseTo(
      output.output,
      matrixFrom([
        [1, 2, 3, 4],
        [1, 2, 5, 6],
      ]),
      0,
    );
  });

  it("does not attend to future keys and values", () => {
    const output = runSelfAttentionHead(
      matrixFrom([[1], [0]]),
      matrixFrom([[0], [10]]),
      matrixFrom([[1], [5]]),
      1,
      1,
    );

    expectMatrixCloseTo(output.output, matrixFrom([[1], [3]]));
  });

  it("flattens each head's causal softmax weights into contiguous column blocks", () => {
    // Zero Q and K make every visible key equally relevant, so each query row
    // is a uniform distribution over its causal window: [1], then [1/2, 1/2],
    // then [1/3, 1/3, 1/3] - identical for both heads.
    const zeros = matrixFrom([
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]);

    const { softmaxOutput } = runSelfAttentionHead(zeros, zeros, zeros, 2, 2);

    // softmaxOutput is vectors x (vectors * headCount) = 3 x 6.
    // Row i holds head 0's weights over keys 0..i, then head 1's, each block
    // `vectors` (3) columns wide with the masked future keys left at zero.
    const third = 1 / 3;
    expectMatrixCloseTo(
      softmaxOutput,
      matrixFrom([
        [1, 0, 0, 1, 0, 0],
        [0.5, 0.5, 0, 0.5, 0.5, 0],
        [third, third, third, third, third, third],
      ]),
    );
  });

  it("reads each head's own softmax block when head dimensionality differs from sequence length", () => {
    // headDimensionsCount (1) deliberately != vectors (2): every previous
    // output-level test used headDim == vectors, which hides the case where
    // the flattened read offset must scale by `vectors`, not `headDim`.
    //
    // Head 0 (column 0) gets zero Q/K -> uniform [0.5, 0.5] over the two keys.
    // Head 1 (column 1) gets logits [0, ln2] -> softmax [1/3, 2/3].
    const inputQ = matrixFrom([
      [0, 0],
      [0, 1],
    ]);
    const inputK = matrixFrom([
      [0, 0],
      [0, Math.log(2)],
    ]);
    const inputV = matrixFrom([
      [1, 10],
      [2, 20],
    ]);

    const { output } = runSelfAttentionHead(inputQ, inputK, inputV, 2, 1);

    // Row 0 sees only itself -> equals inputV row 0.
    // Row 1, head 0:  0.5*1  + 0.5*2  = 1.5
    // Row 1, head 1: (1/3)*10 + (2/3)*20 = 50/3  (NOT 0.5*10 + (1/3)*20 = 11.667,
    //   which is what reading head 0's block at the wrong offset would give).
    // 50/3 isn't exactly representable in f32, so loosen precision past the
    // default to tolerate ~2e-6 of accumulation noise (the other entries are
    // exact, the bug this guards against is off by whole units).
    expectMatrixCloseTo(
      output,
      matrixFrom([
        [1, 10],
        [1.5, 50 / 3],
      ]),
      5,
    );
  });
});

describe("runSelfAttentionMechanism", () => {
  it("projects Q/K/V once, splits heads by feature columns, then applies one shared output projection", () => {
    const twoHeadAttention: AttentionWeights = {
      Q: matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      K: matrixFrom([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      V: matrixFrom([
        [1, 0, 10, 0],
        [3, 0, 30, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]),
      out: matrixFrom([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
      ]),
    };

    const output = runSelfAttentionMechanism(input, 2, twoHeadAttention);

    expectMatrixCloseTo(
      output.output,
      matrixFrom([
        [1, 10, 0, 0],
        [2, 20, 0, 0],
      ]),
    );
  });
});
