In this repo I'm trying to build a very basic LLM from the ground up.

When asked question always give HINTS but don't spell out the answer.

The goal of this project is for me to learn exactly how the systems work by implementing everything myself.

Do NOT edit any files that implement inference or training. Exception is test cases; you are allowed to write testcases for me to do TDD against.

## Coding convention
Any code you write; do not add any comments unless explicitly instructed

## Commands
- `pnpm test` — run the CPU test suite (vitest, `*.test.ts`)
- `pnpm test:gpu:all` — run every GPU test (`*.gpu-test.ts`) via bun + preload; needs a real GPU adapter
- `pnpm test:gpu` — bun GPU runner with preload; append a path to run a single `.gpu-test.ts`
- `pnpm typecheck` — type-check the project with tsc

GPU tests (`*.gpu-test.ts`) only run under bun, not vitest.