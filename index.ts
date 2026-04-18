import { program } from "commander";
import { runLlm } from "./llm.ts";
import { toyWeights } from "./weights/toy_weights/toyWeights.ts";

program
  .name("llm")
  .argument("<input...>", "raw input to complete")
  .action((inputSeperated: string[]) => {
    console.log(runLlm(inputSeperated.join(" "), toyWeights));
  });

program.parse();
