import { readRawValidationData } from "../model/model-io.ts";
import type { Model } from "../model/model-types.ts";
import { sum } from "../shared/math.ts";
import { getSequenceLoss } from "./getSequenceLoss.ts";
import { prepareExampleData } from "./prepareExampleData.ts";

export const runValidationCheck = async (
  modelName: string,
  model: Model,
): Promise<number> => {
  const validationData = prepareExampleData(
    readRawValidationData(modelName),
    model.vocabulary,
    model.trainingMaskSeparator ?? null,
  );

  const totals = await validationData.reduce(
    async (accP, example) => {
      const acc = await accP;

      const { unmaskedTokenCount, outputLosses } = getSequenceLoss(
        example,
        model,
      );

      const summedLoss = sum(outputLosses);

      return {
        summedLoss: acc.summedLoss + summedLoss,
        unmaskedTokenCount: acc.unmaskedTokenCount + unmaskedTokenCount,
      };
    },
    Promise.resolve({
      summedLoss: 0,
      unmaskedTokenCount: 0,
    }),
  );

  return totals.summedLoss / totals.unmaskedTokenCount;
};
