import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { parseTrainingData } from "./prepareTrainingData.ts";

describe("parseTrainingData", () => {
  const vocab = ["hello", " world", "my", " name", " is", " beer"];

  it("parses separated training sequences into token arrays", () => {
    expect(
      parseTrainingData(
        `hello world ${END_OF_SEQUENCE_TOKEN}my name is beer`,
        vocab,
      ),
    ).toEqual([
      ["hello", " world"],
      ["my", " name", " is", " beer"],
    ]);
  });

  it("ignores an empty sequence after a trailing end-of-sequence token", () => {
    expect(
      parseTrainingData(`hello world ${END_OF_SEQUENCE_TOKEN}`, vocab),
    ).toEqual([["hello", " world"]]);
  });
});
