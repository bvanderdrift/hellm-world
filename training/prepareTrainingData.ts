import { readFileSync } from "fs";
import { join } from "path";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { tokenize } from "../shared/tokenizer.ts";

export const prepareTrainingData = (
  model: string,
  vocabulary: string[],
): string[][] => {
  const modelTrainingDataFile = join(
    import.meta.dirname,
    "../weights",
    model,
    "_training_data.txt",
  );
  const modelTrainingDataContent = readFileSync(
    modelTrainingDataFile,
  ).toString();

  return parseTrainingData(modelTrainingDataContent, vocabulary);
};

export const parseTrainingData = (
  fileContent: string,
  vocabulary: string[],
): string[][] => {
  const sequences = fileContent
    .split(END_OF_SEQUENCE_TOKEN)
    // Remove empty lines
    .filter((sequence) => !!sequence);

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
