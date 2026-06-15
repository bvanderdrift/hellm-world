In this repo I'm trying to build a very basic LLM from the ground up.

The goal of this project is for me to learn exactly how the systems work by implementing everything myself.

## Answering inference & training question
In this case always give HINTS but don't spell out the answer when related to inference or training.

Examples:
* Why is the test failing?
* What am I doing wrong?

## Answering understanding questions
Always give straight up and concise answers to question that increase understanding or are related to frameworks.

Examples:
* How does weight decay work?
* How do I implement workgroup dispatch in TypeGPU?

## Editing files
Do NOT functionally edit or implement inference or training logic.

Exception: (not limited to)
* test cases; you are allowed to write testcases for me to do TDD against.
* benchmarking scripts
* CLI harnass
* moving code around without changing functionality

## Coding convention
Any code you write; do not add any comments unless explicitly instructed

## Commands
- `pnpm test` — run the CPU test suite (vitest, `*.test.ts`)
- `pnpm test:gpu:all` — run every GPU test (`*.gpu-test.ts`) via bun + preload; needs a real GPU adapter
- `pnpm test:gpu` — bun GPU runner with preload; append a path to run a single `.gpu-test.ts`
- `pnpm typecheck` — type-check the project with tsc

GPU tests (`*.gpu-test.ts`) only run under bun, not vitest.