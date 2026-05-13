import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { tokenize } from "../shared/tokenizer.ts";
import { readRawTrainingData } from "../model/model-io.ts";
import type { TrainingExample } from "./doSingleTrainingPass.ts";

export const prepareTrainingData = (
  modelName: string,
  vocabulary: string[],
  maskSeparator: string | null,
): TrainingExample[] => {
  const modelTrainingDataContent = readRawTrainingData(modelName);

  const sequences = parseTrainingData(modelTrainingDataContent, vocabulary);

  return sequences.map(
    (sequence): TrainingExample => ({
      sequence,
      maskBeforeIndex:
        maskSeparator !== null ? sequence.indexOf(maskSeparator) : null,
    }),
  );
};

export const parseTrainingData = (
  fileContent: string,
  vocabulary: string[],
): string[][] => {
  const sequences = fileContent
    .split(END_OF_SEQUENCE_TOKEN)
    // Remove empty lines
    .filter((sequence) => !!sequence)
    // Add back the EOS token we split on
    .map((sequence) => `${sequence}${END_OF_SEQUENCE_TOKEN}`);

  return sequences.map((sequence) =>
    tokenize(
      sequence
        // this only replaces the first \n
        .replace("\\n", "")
        .trim(),
      vocabulary,
    ),
  );
};
