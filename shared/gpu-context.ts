import tgpu from "typegpu";
import { setupGlobals } from "bun-webgpu";
setupGlobals();

export const gpuContext = await tgpu.init();
