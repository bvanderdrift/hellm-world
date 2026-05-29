import { describe, expect, it } from "vitest";
import { END_OF_SEQUENCE_TOKEN } from "../shared/const.ts";
import { parseExampleData, prepareExampleData } from "./prepareExampleData.ts";

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

  it("filters out blank lines between training examples", () => {
    const mathVocab = ["1", "2", "3", "+", "=", END_OF_SEQUENCE_TOKEN];

    const result = parseExampleData(
      `1+2=3${END_OF_SEQUENCE_TOKEN}\n\n3+1=2${END_OF_SEQUENCE_TOKEN}\n`,
      mathVocab,
    );

    expect(result).toEqual([
      ["1", "+", "2", "=", "3", END_OF_SEQUENCE_TOKEN],
      ["3", "+", "1", "=", "2", END_OF_SEQUENCE_TOKEN],
    ]);
  });

  it("filters out a trailing newline after the last EOS", () => {
    const mathVocab = ["1", "2", "+", "=", END_OF_SEQUENCE_TOKEN];

    const result = parseExampleData(
      `1+2=2${END_OF_SEQUENCE_TOKEN}\n`,
      mathVocab,
    );

    expect(result).toEqual([
      ["1", "+", "2", "=", "2", END_OF_SEQUENCE_TOKEN],
    ]);
  });
});

describe("prepareExampleData", () => {
  const mathVocab = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "+", "=", END_OF_SEQUENCE_TOKEN,
  ];

  it("never produces a sequence where every token is masked", () => {
    const input =
      `1+2=3${END_OF_SEQUENCE_TOKEN}\n\n3+1=4${END_OF_SEQUENCE_TOKEN}\n`;

    const examples = prepareExampleData(input, mathVocab, "=");

    for (const example of examples) {
      const unmaskedCount = example.sequence.filter((_, index) => {
        if (
          example.maskBeforeIndex !== null &&
          index < example.maskBeforeIndex
        ) {
          return false;
        }
        const nextToken = example.sequence[index + 1];
        return nextToken !== undefined && mathVocab.includes(nextToken);
      }).length;

      expect(unmaskedCount).toBeGreaterThan(0);
    }
  });
});
