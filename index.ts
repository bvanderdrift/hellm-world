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
  .action((opts: { steps: number }) => {
    doTrainingLoopAndStoreCheckpoint("toy_model", opts.steps);
  });

program.parse();
