/**
 * Worker for the parallel `addy-new` correctness map.
 *
 * Each worker loads the requested checkpoint once, then endlessly samples
 * random (a, b) pairs, tests them, and posts batches of results back to the
 * root process. The root owns the grid/file/chart; workers only ever produce
 * `{ a, b, correct }` tuples.
 *
 * This is an independent script: it only imports already-exported inference
 * helpers, it does not modify any inference/training code.
 */

import { type Model } from "../../../model/model-types.ts";
import { tokenize } from "../../../shared/tokenizer.ts";
import { getRawVector } from "../../../shared/matrices/matrices.ts";
import { END_OF_SEQUENCE_TOKEN } from "../../../shared/const.ts";
import { llmForwardPassByTokens } from "../../../running/llm.ts";
import { getHighestValueIndex } from "../../../running/llm-shared.ts";
import { getCheckpointModel } from "../../../model/model-checkpoint-io.ts";

import { GRID_SIZE, MODEL_NAME } from "./correctness-grid.ts";

// prevents TS errors
declare var self: Worker;

export type WorkerInput = { checkpointId: number };

export type WorkerResult = { a: number; b: number; correct: boolean };

export type WorkerOutput = { type: "results"; results: WorkerResult[] };

const postMessage = (message: WorkerOutput) => self.postMessage(message);

/** Greedy argmax of the next-token logits for a given token sequence. */
const nextToken = (tokens: string[], model: Model): string => {
  const { embeddings: unembeddedState } = llmForwardPassByTokens(
    tokens,
    model,
    false,
  );
  const logits = getRawVector(unembeddedState, unembeddedState.vectors - 1);
  const index = getHighestValueIndex(logits);
  const token = model.vocabulary[index];
  if (!token) {
    throw new Error(`Failed to find token at index ${index}`);
  }
  return token;
};

/**
 * Returns true iff the model produces exactly `expected` digits followed by
 * <EOS>. Stops generating as soon as a token diverges from the expected answer.
 */
const isCorrect = (a: number, b: number, model: Model): boolean => {
  const input = `${a}+${b}=`;
  const inputTokens = tokenize(input, model.vocabulary);
  const expected = String(a + b);

  const generated: string[] = [];

  // expected.length digit tokens, then one <EOS>. Small cap guards runaways.
  const maxTokens = expected.length + 1;

  for (let step = 0; step < maxTokens; step++) {
    const token = nextToken([...inputTokens, ...generated], model);

    if (step < expected.length) {
      // Still expecting a digit; it must match exactly.
      if (token !== expected[step]) {
        return false;
      }
    } else {
      // All digits matched; the sequence must now terminate.
      return token === END_OF_SEQUENCE_TOKEN;
    }

    generated.push(token);
  }

  return false;
};

self.onmessage = async (event: MessageEvent<WorkerInput>) => {
  const { checkpointId } = event.data;
  const model = getCheckpointModel(MODEL_NAME, checkpointId);

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const a = Math.floor(Math.random() * GRID_SIZE);
    const b = Math.floor(Math.random() * GRID_SIZE);

    // Post each result the moment it's known so the root's progress and counts
    // advance per test rather than in bursts.
    postMessage({ type: "results", results: [{ a, b, correct: isCorrect(a, b, model) }] });

    // Yield to the event loop so the posted message actually flushes to the
    // root before we start the next (CPU-bound) test.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
};
