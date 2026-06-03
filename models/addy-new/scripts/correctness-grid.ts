/**
 * Shared grid storage for the `addy-new` correctness map.
 *
 * The grid is a GRID_SIZE x GRID_SIZE array of cells, one per (a, b) pair,
 * each encoding whether the model got that addition right. It is persisted as
 * a single raw byte file per checkpoint (`correctness_grid.bin`) and is the
 * hand-off point between the testing script (which writes it) and the charting
 * script (which reads it).
 */

import { existsSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";

import { getModelFolderPath } from "../../../model/model-io.ts";
import { getCheckpointFolderPath } from "../../../model/model-checkpoint-io.ts";

export const MODEL_NAME = "addy-new";

// Result grid is 10_000 x 10_000 = 100M cells.
export const GRID_SIZE = 10_000;
export const CELL_COUNT = GRID_SIZE * GRID_SIZE;

// Cell encoding (acts as the requested (boolean | null)[][]).
export const NULL = 0;
export const FALSE = 1;
export const TRUE = 2;

/** Absolute path to this checkpoint's folder, where all artifacts live. */
export const checkpointFolderPath = (checkpointId: number): string =>
  getCheckpointFolderPath(getModelFolderPath(MODEL_NAME), checkpointId);

/** Single per-checkpoint file holding the raw grid bytes (overwritten on save). */
export const gridPath = (checkpointId: number): string =>
  join(checkpointFolderPath(checkpointId), "correctness_grid.bin");

/** Persist the raw Uint8Array grid, overwriting the checkpoint's single file. */
export const saveGrid = (grid: Uint8Array, checkpointId: number) => {
  writeFileSync(gridPath(checkpointId), grid);
};

/**
 * Load a previously saved grid for this checkpoint, or return a fresh one.
 * `testCount`/`correctCount` are derived from the grid: each non-null cell is a
 * recorded test, each TRUE cell a correct one.
 */
export const loadGrid = (
  checkpointId: number,
): { grid: Uint8Array; testCount: number; correctCount: number } => {
  const path = gridPath(checkpointId);
  if (!existsSync(path)) {
    return { grid: new Uint8Array(CELL_COUNT), testCount: 0, correctCount: 0 };
  }

  const buffer = readFileSync(path);
  if (buffer.length !== CELL_COUNT) {
    throw new Error(
      `Saved grid ${path} has ${buffer.length} bytes, expected ${CELL_COUNT}`,
    );
  }
  const grid = new Uint8Array(buffer);

  let testCount = 0;
  let correctCount = 0;
  for (let i = 0; i < grid.length; i++) {
    const value = grid[i]!;
    if (value === NULL) continue;
    testCount++;
    if (value === TRUE) correctCount++;
  }

  return { grid, testCount, correctCount };
};
