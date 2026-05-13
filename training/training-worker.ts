import type { Model, Weights } from "../model/model-types.ts";
import {
  doSingleTrainingPass,
  type TrainingExample,
} from "./doSingleTrainingPass.ts";

// prevents TS errors
declare var self: Worker;

export type InputMessagePayload = {
  model: Model;
  trainingData: TrainingExample[];
};

export type OutputMessagePayload = {
  averageLoss: number;
  adjustedWeights: Weights;
};

self.onmessage = async (event: MessageEvent<InputMessagePayload>) => {
  const output: OutputMessagePayload = await doSingleTrainingPass(
    event.data.model,
    event.data.trainingData,
  );

  self.postMessage(output);
};
