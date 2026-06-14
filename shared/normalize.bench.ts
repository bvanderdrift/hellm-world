import { normalize } from "./normalize.ts";
import { normalizeOnGpu } from "./normalize-gpu.ts";
import { compareAcrossSizes } from "../bench-harness.ts";

if (import.meta.main) {
  await compareAcrossSizes({
    name: "normalize CPU vs GPU benchmark",
    tolerance: 1e-4,
    cpu: ({ matrix }) => normalize(matrix),
    gpu: ({ buffer }) => {
      normalizeOnGpu(buffer);
      return buffer;
    },
  });
}
