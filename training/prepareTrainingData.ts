import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { tokenize } from "../shared/tokenizer.ts";
import { readRawTrainingData } from "../model/model-io.ts";

export const prepareTrainingData = (
  modelName: string,
  vocabulary: string[],
): string[][] => {
  const modelTrainingDataContent = readRawTrainingData(modelName);

  return parseTrainingData(modelTrainingDataContent, vocabulary);
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
