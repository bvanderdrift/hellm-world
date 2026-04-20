import { program } from "commander";
import { runLlm } from "./llm.ts";

program
  .name("llm")
  .argument("<input...>", "raw input to complete")
  .action((inputSeperated: string[]) => {
    console.log(runLlm(inputSeperated.join(" "), "toy_model"));
  });

program.parse();
