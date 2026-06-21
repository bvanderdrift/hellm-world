export const startKeyboardListening = ({
  onSave,
}: {
  onSave: () => void | Promise<void>;
}) => {
  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.setEncoding("utf8");

  process.stdin.on("data", async (key: string) => {
    if (key === "\u0003") {
      console.log(`Cancel command received... Storing checkpoint and exiting`);
      await onSave();
      process.exit(); // Ctrl+C
    }

    if (key === "s" || key === "S") {
      await onSave();
    }
  });
};
