import type { Model, Weights } from "../../model/model-types.ts";
import {
  doSingleTrainingPass,
  type TrainingExample,
  type TrainingPassOutput,
} from "../doSingleTrainingPass.ts";

// prevents TS errors
declare var self: Worker;

export type InputMessagePayload = {
  model: Model;
  trainingData: TrainingExample[];
};

export type ResultsMessagePayload = {
  type: "results";
} & TrainingPassOutput;

export type OutputMessagePayload =
  | ResultsMessagePayload
  | {
      type: "step-complete";
      durationMs: number;
    };

// just for type-safety
const postMessage = (message: OutputMessagePayload) =>
  self.postMessage(message);

self.onmessage = async (event: MessageEvent<InputMessagePayload>) => {
  const results = await doSingleTrainingPass(
    event.data.model,
    event.data.trainingData,
    (durationMs) => {
      postMessage({ type: "step-complete", durationMs });
    },
  );

  postMessage({
    type: "results",
    ...results,
  });
};
