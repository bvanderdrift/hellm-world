import { program } from "commander";
import { runLlm } from "./running/llm.ts";
import { doTrainingLoopAndStoreCheckpoint } from "./training/training.ts";

program
  .name("llm")
  .command("run")
  .argument("<input...>", "raw input to complete")
  .action((inputSeperated: string[]) => {
    console.log(runLlm(inputSeperated.join(" "), "toy_model"));
  });

program
  .name("train")
  .command("train")
  .option(
    "-s, --steps <steps>",
    "Amount of steps before storing another checkpoint",
  )
  .argument("<model>", "model to run")
  .action((modelName: string, opts: { steps: number }) => {
    doTrainingLoopAndStoreCheckpoint(modelName, opts.steps);
  });

program.parse();
