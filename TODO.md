# TODO

## First decision

- Decide whether this repo is aiming for:
  - a custom educational transformer that you train and run only with your own weight format
  - or a more checkpoint-compatible decoder model that resembles common GPT/LLaMA-style layouts
- Hint: make this choice before serious training work. It changes what "done" means for norms, attention weights, tokenizer format, and checkpoint I/O.

## Minimum path to "train weights, then run them"

- [x] Move model state out of hardcoded source constants.
- Hint: `weights.ts` is acting like an in-repo checkpoint right now. Think about what a saved artifact should contain, and how `llm.ts` should receive it.

- [x] Separate "forward pass" from "generation".
- Hint: one function can return logits for a prompt; a different layer can turn those logits into one or more emitted tokens.

- [ ] Add a real load path for vocab/tokenizer artifacts.
- Hint: your current vocab order is implicit in `tokenizer.ts`. Ask what must stay stable so token id `n` always means the same thing during training and inference.

- [x] Add a decode path, not just encode/tokenize.
- Hint: once the model predicts an id, the runtime should be able to turn that id back into a token/string without depending on source edits.

## If you keep the current custom architecture

- [ ] Write down the architecture contract explicitly.
- Hint: future-you should be able to answer "what exact tensors must a checkpoint contain?" without reading all the code.

- [ ] Add end-to-end tests around the assembled model.
- Hint: unit tests for math blocks are good, but they do not yet prove that a trained checkpoint would survive load -> forward pass -> token selection.

- [ ] Add a small generation loop.
- Hint: greedy one-token prediction is already close; the missing idea is how to append the chosen token and stop at the right time.

## If you want a more standard decoder architecture

- [ ] Revisit normalization first.
- Hint: your current `normalize(...)` is a fixed statistical transform, not a learned module. Standard checkpoints expect norm parameters to live somewhere in the weight format.

- [ ] Revisit attention parameterization.
- Hint: compare your per-head `V.down` / `V.up` structure with the more common "project Q/K/V, combine heads, then apply one shared output projection" pattern.

- [ ] Revisit the MLP activation.
- Hint: `ReLU` works, but it is not the usual choice in modern LLMs.

- [ ] Revisit positional encoding.
- Hint: sinusoidal encoding is valid, but it is not the default choice in many current decoder-only LLMs.

## Runtime quality-of-life

- [ ] Make CLI inference robust to unknown input or bad artifacts.
- Hint: the tokenizer CLI already catches errors; the LLM CLI should eventually have equally clear failure modes.

- [ ] Add checkpoint validation beyond shape checks.
- Hint: a tensor can have the right dimensions and still be semantically wrong because vocab order, head ordering, or projection layout drifted.

- [ ] Add at least one golden-path test.
- Hint: choose tiny weights where you can predict the next token by hand and assert the full pipeline result.

## Efficiency later

- [ ] Consider a KV cache only after correctness is locked down.
- Hint: reprocessing the whole prompt each step is acceptable for a tiny educational model; caching matters more once the forward path is stable.
