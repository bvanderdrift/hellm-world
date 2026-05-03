import { program } from "commander";
import { input, number } from "@inquirer/prompts";
import { runLlm } from "./running/llm.ts";
import { doTrainingLoopAndStoreCheckpoint } from "./training/training.ts";
import { getLatestCheckpointModel, writeNewModel } from "./model/model-io.ts";
import { decodeVocab, initializeModel } from "./model/model-initialize.ts";
import { describeModelToConsole } from "./model/model-helpers.ts";
import { writeLossChart } from "./scripts/chart-loss.ts";

program
  .name("llm")
  .command("run")
  .argument("<model>", "model to run")
  .argument("<input...>", "raw input to complete")
  .action((modelName: string, inputSeperated: string[]) => {
    const tokenGenerator = runLlm(inputSeperated.join(" "), modelName);

    for (const token of tokenGenerator) {
      process.stdout.write(token);
    }
    process.stdout.write("\n");
  });

program
  .name("train")
  .command("train")
  .argument("<model>", "model to run")
  .option(
    "-s, --steps <steps>",
    "Amount of steps before storing another checkpoint",
  )
  .action((modelName: string, opts: { steps: number }) => {
    doTrainingLoopAndStoreCheckpoint(modelName, opts.steps);
  });

program
  .name("init-model")
  .command("init-model")
  .action(async () => {
    const modelName = await input({
      message: "Model name",
      required: true,
      pattern: /^[a-zA-Z0-9_-]+$/,
      patternError: "Only alphanumerical characters, - and _ are allowed",
    });

    const headsCount = await number({
      message: "Attention Head Count",
      required: true,
      default: 8,
      min: 1,
      step: 1,
    });

    const hiddenDimensionCount = await number({
      message: "Hidden Dimension Count",
      required: true,
      default: 512,
      min: 16,
      step: 1,
    });

    const transformerCount = await number({
      message: "Transformer Count",
      required: true,
      default: 6,
      min: 1,
      step: 1,
    });

    const vocabularySingleString = await input({
      message: "comma-seperated vocabulary list",
      required: true,
      validate(value) {
        const tokens = decodeVocab(value);

        if (!tokens.length) {
          return "No valid tokens found";
        }

        const tokensUnique = new Set(tokens);

        if (tokensUnique.size !== tokens.length) {
          return "Duplicate tokens found";
        }

        return true;
      },
    });

    const vocabulary = decodeVocab(vocabularySingleString);

    const newModel = initializeModel({
      headsCount,
      hiddenDimensionCount,
      transformerCount,
      vocabulary,
    });

    writeNewModel(modelName, newModel);

    describeModelToConsole(newModel);

    console.log(`\nModel "${modelName}" written to models folder`);
  });

program
  .name("describe")
  .command("describe")
  .argument("<model>", "model to describe")
  .action((modelName: string, opts: { steps: number }) => {
    const model = getLatestCheckpointModel(modelName).model;

    describeModelToConsole(model);
  });

program
  .name("chart-loss")
  .command("chart-loss")
  .argument("<model>", "model to chart")
  .argument("[checkpoint]", "checkpoint file", "checkpoint_000007.json")
  .action(async (modelName: string, checkpointName: string) => {
    console.log(await writeLossChart(modelName, checkpointName));
  });

program.parse();
