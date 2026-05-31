import type { LossRecord } from "./loss-weighted-sampling.ts";

export const MAX_TRAINING_DATA_PER_PASS = 100;

export type SamplerState =
  | {
      type: "uniform";
    }
  | {
      type: "loss-weighted";
      lossRecord: LossRecord;
    };
