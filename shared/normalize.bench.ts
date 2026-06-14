import { normalize } from "./normalize.ts";
import { normalizeOnGpu } from "./normalize-gpu.ts";
import { compareAcrossSizes } from "../bench-harness.ts";
import { createMatrixBuffer } from "./matrices-gpu.ts";

if (import.meta.main) {
  await compareAcrossSizes({
    name: "normalize CPU vs GPU benchmark",
    tolerance: 1e-4,
    setup: (size) =>
      createMatrixBuffer({
        vectors: size.vectors,
        dimensions: size.dimensions,
      }),
    cpu: ({ matrix }) => normalize(matrix),
    gpu: ({ buffer }, output) => {
      normalizeOnGpu(buffer, output);
      return output;
    },
  });
}
