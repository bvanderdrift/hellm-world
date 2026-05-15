import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { parseExampleData } from "./prepareExampleData.ts";

describe("parseTrainingData", () => {
  const vocab = [
    "hello",
    " world",
    "my",
    " name",
    " is",
    " beer",
    " ",
    END_OF_SEQUENCE_TOKEN,
  ];

  it("parses separated training sequences into token arrays", () => {
    expect(
      parseExampleData(
        `hello world ${END_OF_SEQUENCE_TOKEN}my name is beer`,
        vocab,
      ),
    ).toEqual([
      ["hello", " world", " ", END_OF_SEQUENCE_TOKEN],
      ["my", " name", " is", " beer", END_OF_SEQUENCE_TOKEN],
    ]);
  });

  it("ignores an empty sequence after a trailing end-of-sequence token", () => {
    expect(
      parseExampleData(`hello world ${END_OF_SEQUENCE_TOKEN}`, vocab),
    ).toEqual([["hello", " world", " ", END_OF_SEQUENCE_TOKEN]]);
  });

  it("keeps whitespace before EOS when the training content includes it", () => {
    expect(
      parseExampleData(`hello ${END_OF_SEQUENCE_TOKEN}`, [
        "hello",
        " ",
        END_OF_SEQUENCE_TOKEN,
      ]),
    ).toEqual([["hello", " ", END_OF_SEQUENCE_TOKEN]]);
  });

  it("does not add whitespace before EOS when the training content omits it", () => {
    expect(
      parseExampleData(`hello${END_OF_SEQUENCE_TOKEN}`, [
        "hello",
        " ",
        END_OF_SEQUENCE_TOKEN,
      ]),
    ).toEqual([["hello", END_OF_SEQUENCE_TOKEN]]);
  });
});
