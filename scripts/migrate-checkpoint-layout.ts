/**
 * One-off migration: convert the "old" flat per-model checkpoint layout into the
 * "new" folder-per-checkpoint layout with static filenames.
 *
 *   checkpoint_000034.bin                    -> checkpoint_000034/weights.bin
 *   checkpoint_000034_training_state.json    -> checkpoint_000034/training_state.json
 *   checkpoint_000034_loss.png               -> checkpoint_000034/loss.png
 *   checkpoint_000034_struggles.txt          -> checkpoint_000034/struggles.txt
 *   correctness_grid_54.bin                  -> checkpoint_000054/correctness_grid.bin
 *   correctness_54_18000.png                 -> checkpoint_000054/correctness_18000.png
 *
 * Everything else (model-level metadata, training data, distributions, test
 * outputs, scripts/, the legacy _training_history.json) stays at the model root.
 *
 * Operates on the filesystem directly so it also moves git-ignored / untracked
 * artifacts. Dry-run by default; pass --apply to actually move files.
 *
 * Usage:
 *   bun run scripts/migrate-checkpoint-layout.ts            # dry run
 *   bun run scripts/migrate-checkpoint-layout.ts --apply    # perform moves
 */

import {
  existsSync,
  mkdirSync,
  readdirSync,
  renameSync,
  statSync,
} from "fs";
import { join } from "path";

const MODELS_DIR = join(import.meta.dirname, "..", "models");
const PAD = (n: number) => n.toString().padStart(6, "0");

type Move = { from: string; to: string; checkpoint: string; kind: string };

/**
 * Map a single root-level entry name to its destination relative to the model
 * folder, or `null` if it should stay where it is.
 */
const planEntry = (
  name: string,
): { checkpoint: string; target: string; kind: string } | null => {
  let m: RegExpMatchArray | null;

  if ((m = name.match(/^(checkpoint_\d{6})\.bin$/))) {
    return { checkpoint: m[1]!, target: "weights.bin", kind: "weights" };
  }
  if ((m = name.match(/^(checkpoint_\d{6})_training_state\.json$/))) {
    return {
      checkpoint: m[1]!,
      target: "training_state.json",
      kind: "training_state",
    };
  }
  if ((m = name.match(/^(checkpoint_\d{6})_loss\.png$/))) {
    return { checkpoint: m[1]!, target: "loss.png", kind: "loss" };
  }
  if ((m = name.match(/^(checkpoint_\d{6})_struggles\.txt$/))) {
    return { checkpoint: m[1]!, target: "struggles.txt", kind: "struggles" };
  }
  if ((m = name.match(/^correctness_grid_(\d+)\.bin$/))) {
    return {
      checkpoint: `checkpoint_${PAD(Number(m[1]))}`,
      target: "correctness_grid.bin",
      kind: "correctness_grid",
    };
  }
  if ((m = name.match(/^correctness_(\d+)_(\d+)\.png$/))) {
    return {
      checkpoint: `checkpoint_${PAD(Number(m[1]))}`,
      target: `correctness_${m[2]}.png`,
      kind: "correctness_png",
    };
  }

  return null;
};

const planModel = (modelFolderPath: string): Move[] => {
  const moves: Move[] = [];

  for (const name of readdirSync(modelFolderPath)) {
    const fullPath = join(modelFolderPath, name);
    if (!statSync(fullPath).isFile()) continue; // skip checkpoint_* folders, scripts/

    const plan = planEntry(name);
    if (!plan) continue;

    moves.push({
      from: fullPath,
      to: join(modelFolderPath, plan.checkpoint, plan.target),
      checkpoint: plan.checkpoint,
      kind: plan.kind,
    });
  }

  return moves;
};

const main = () => {
  const apply = process.argv.includes("--apply");

  const modelNames = readdirSync(MODELS_DIR).filter((name) =>
    existsSync(join(MODELS_DIR, name, "_metadata.json")),
  );

  let total = 0;
  let conflicts = 0;
  const kindCounts: Record<string, number> = {};

  for (const modelName of modelNames) {
    const modelFolderPath = join(MODELS_DIR, modelName);
    const moves = planModel(modelFolderPath);
    if (moves.length === 0) continue;

    console.log(`\n=== ${modelName} (${moves.length} files) ===`);

    for (const move of moves) {
      kindCounts[move.kind] = (kindCounts[move.kind] ?? 0) + 1;
      total++;

      const rel = (p: string) => p.slice(modelFolderPath.length + 1);

      if (existsSync(move.to)) {
        conflicts++;
        console.log(`  ⚠️  SKIP (target exists): ${rel(move.from)}`);
        continue;
      }

      console.log(`  ${rel(move.from)}  ->  ${rel(move.to)}`);

      if (apply) {
        mkdirSync(join(modelFolderPath, move.checkpoint), { recursive: true });
        renameSync(move.from, move.to);
      }
    }
  }

  console.log(`\n--- summary ---`);
  for (const [kind, count] of Object.entries(kindCounts).sort()) {
    console.log(`  ${kind}: ${count}`);
  }
  console.log(`  total: ${total}  conflicts(skipped): ${conflicts}`);
  console.log(
    apply
      ? `\n✅ Applied. ${total - conflicts} files moved.`
      : `\nDry run only. Re-run with --apply to perform the moves.`,
  );
};

main();
