import { program } from "commander";
import { runLlm } from "./running/llm.ts";
import { doTrainingLoopAndStoreCheckpoint } from "./training/training.ts";

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

program.parse();
