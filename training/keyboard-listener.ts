import type { StateStore } from "./training-state.ts";

export const startKeyboardListening = (store: StateStore) => {
  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.setEncoding("utf8");

  process.stdin.on("data", (key: string) => {
    if (key === "\u0003") {
      console.log(`Cancel command received... Storing checkpoint and exiting`);
      store.writeNewCheckpoint();
      process.exit(); // Ctrl+C
    }

    if (key === "s" || key === "S") {
      store.writeNewCheckpoint();
    }
  });
};
