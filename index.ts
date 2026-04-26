import { program } from "commander";
import { runLlm } from "./running/llm.ts";
import { doTrainingLoopAndStoreCheckpoint } from "./training/training.ts";

program
  .name("llm")
  .command("run")
  .argument("<model>", "model to run")
  .argument("<input...>", "raw input to complete")
  .action((modelName: string, inputSeperated: string[]) => {
    console.log(runLlm(inputSeperated.join(" "), modelName));
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
