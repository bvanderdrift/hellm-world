import { program } from "commander";
import { tokenize } from "./tokenizer.ts";

program
  .name("tokenize")
  .argument("<input...>", "raw input to tokenize")
  .action((inputSeperated: string[]) => {
    try {
      const input = inputSeperated.join(" ");
      console.log(tokenize(input));
    } catch (error) {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    }
  });

program.parse();
