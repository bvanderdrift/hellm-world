# Speeding up `multiplyMatricesOnGPU`

Notes-to-self for improving the GPU matmul in `matrices-gpu.ts`, ordered by **impact ÷ complexity** (do the top ones first). Numbers are from the M1 7-core GPU via the tuning sweep; the roofline bench (`scripts/matmul-roofline-bench.ts`) compares against an **edge-safe** tuned-SGEMM ceiling — one that handles arbitrary (ragged) sizes, not just perfectly tile-aligned ones — so the target is something you could actually ship.

Baseline today: **~175 GFLOP/s.** The realistic ceiling depends on the shape:

- **Compute-heavy layers** (proj `MxKxN = Mx256x256`, mlp `Mx256x1024` / `Mx1024x256`) with a real batched `M`: ceiling ~600–660 GFLOP/s, current reaches **~27–33%** → roughly **3.5× headroom**. Most of it comes from the first three items below.
- **Logits** (`Mx256x13`): ceiling is only **~35–85 GFLOP/s** and *no kernel tuning fixes it much* — see the "skinny N" note below.

The old "720 GFLOP/s" figure assumed a perfect 2048³ square with zero edge handling; the edge-safe number is a bit lower (~660) and is the honest target.

---

## 0. Gotcha to know before you refactor indices (trivial, but a correctness trap)

TypeGPU compiles `/` as **floating-point** division (JS semantics), not integer division. So `(tid / 16) * 4` becomes `(f32(tid) / 16.0) * 4.0` and the fractional part leaks into whatever index you build from it. The current kernel mostly dodges this by wrapping whole index expressions in implicit `u32(...)`, but the moment you compute a row/col group like `floor(tid / something)` and then multiply, you must floor it first:

```ts
const rowGroup = d.u32(input.tid / d.u32(16)); // floor to integer FIRST
const row = rowGroup * 4; // now safe to multiply
```

`%` stays integer, so `(tid % 16) * 4` is already fine. Keep this in mind for every refactor below — a wrong result here looks like a ~20-magnitude error, not a crash.

---

## 1. Accumulate into registers, not a dynamically-indexed local array  ⭐ highest impact

**Impact: ~2.4× (180 → ~440). Complexity: low.**

Right now the inner loop accumulates into `sums` (`matrices-gpu.ts:85`), a `d.arrayOf(d.f32, 16)` indexed by a *computed* `summerIndex`. WGSL compilers can't keep a dynamically-indexed array in registers, so it gets spilled to **private memory** — every `+=` is a load+store to memory instead of a register op. That alone caps you well below peak.

**Fix:** give each accumulator its own named `let`. For a 4×4 micro-tile that's 4 `vec4f` accumulators (`acc0..acc3`), one per output row, each holding 4 columns. Index them with *constant* literals only (`acc0`, `acc1`, …), never a loop variable. Same rule applies to the per-thread A/B values you load into registers.

The tell: dump the WGSL (`--dump-wgsl` via the bench harness) and look for `var sums: array<f32, 16>` declared inside the function — that's the spill.

---

## 2. Vectorize the math with `vec4f` + `fma`  ⭐

**Impact: large (compounds with #1; vectorizing the shared reads took ~570 → ~720). Complexity: low–medium.**

The device's FP throughput probe hits its number using `vec4f` fused multiply-add (`scripts/gpu-roofline-bench.ts`). Scalar `a * b` accumulation leaves a lot of that on the floor. Restructure the inner product as an **outer product**: for each `k`, load one column of A (a few scalars) and one row of B (a `vec4f`), then

```ts
acc = std.fma(d.vec4f(a, a, a, a), bRow, acc); // one fma updates 4 columns
```

Two sub-steps, in order of payoff:

- **Store the B tile as `vec4f` in workgroup memory** so reading `bRow` is a single vectorized load instead of 4 scalar loads. (~430 → ~530.)
- **Store the A tile transposed as `vec4f`** (`As[k][m]` layout) so the 4–8 A values a thread needs for a given `k` come from 1–2 `vec4f` loads instead of one scalar load each. This was the single biggest jump (~570 → ~720) because A reads were the last scalar bottleneck left in the inner loop.

Note `d.vec4f(x)` does **not** splat like WGSL's `vec4f(x)` — pass all four components (`d.vec4f(x, x, x, x)`) or you'll only fill `.x`.

---

## 3. Tune the register-blocking shape (per-thread micro-tile)

**Impact: moderate (~530 → ~570). Complexity: low once #1 and #2 are in.**

Each thread should compute a small dense block of outputs (a "micro-tile"), not one element. More outputs per thread = more arithmetic per shared-memory load (higher arithmetic intensity) and more independent FMAs to hide latency.

On this 7-core M1 the sweet spot was an **8×4 micro-tile with 128 threads per workgroup** (64×64 output tile). Worth sweeping:

- micro-tile: 4×4, 8×4, 8×8
- workgroup size: 64 / 128 / 256 threads

Counter-intuitive result worth remembering: 8×8 was *slower* here — but only because that version used an indexed accumulator array (see #1). With named registers the picture changes, so re-measure rather than trusting intuition. Bigger isn't automatically better: too many registers per thread drops occupancy.

---

## 4. Keep a fast path, but don't break correctness on ragged sizes

**Impact: enabler, not raw speed. Complexity: medium.**

The fast kernels above assume sizes that are multiples of the block (the model's real shapes may not be). Two options:

- **Branchless bounds via padding:** allocate matrices rounded up to the block size and zero-pad. Keeps the inner loop branch-free (fastest), costs a little memory and a pad step.
- **Specialized tail kernel:** one no-bounds-check kernel for full tiles plus a slower bounds-checked kernel for edge tiles.

The current kernel bounds-checks *every* element load (`matrices-gpu.ts:106`, `:116`), which adds per-iteration overhead to the common case. Hoisting that out of the hot loop matters more once the loop itself is fast.

---

## 4b. Skinny-N layers (logits, `N = vocab = 13`) need a different shape

**Impact: large *for that layer*, which is run every token. Complexity: medium.**

The logits matmul has `N = 13`. A 64-wide column tile therefore wastes ~80% of its columns, and `vec4` column loads straddle the `N=13` edge — that's why its ceiling (~35–85 GFLOP/s) is ~10× below the other layers regardless of how good the inner loop is. This is a *shape* problem, not an instruction problem.

Options, roughly in order: shrink the N-tile (e.g. `BN=16`) so you're not paying for 64 columns you don't have; or for very small N, drop register-blocking on N entirely and assign one thread per output column with a long-K accumulation. Measure — for `N=13` the simplest "few columns, long K" kernel may beat the fancy one. Don't over-invest until the big layers (item 1–3) are done, but remember this layer is on the hot path of every generated token.

## 5. Vectorize the *global* loads too (true coalesced reads)

**Impact: small-to-moderate, situational. Complexity: medium.**

In #2 the `vec4f` lived in workgroup memory but the global buffer is still `array<f32>`, so the global→shared copy is still scalar. Typing the storage buffers as `array<vec4f>` lets the load from global memory itself be a 16-byte vectorized, coalesced transaction. Requires changing the bind-group layout and the buffer/struct definitions, and only helps if you're not already compute-bound — measure before/after, it may be in the noise on large N.

---

## 6. Things that did *not* help here (so you can skip or deprioritize)

- **Bigger K-strip (`BK`):** going from `BK=8` to `BK=16` was flat-to-slightly-worse on this GPU. Low effort to try, but don't expect a win.
- **Double-buffering the shared tiles** (overlap the next load with the current compute, removing one barrier): classic technique, but with only ~25%→60% already reached, the barriers weren't the dominant cost. Higher complexity, uncertain payoff — leave for last.
- **Subgroup / SIMD-group ops** (`std.subgroup*`): powerful but fiddly and device-dependent; not worth it until everything above is exhausted.

---

## How to measure

- `scripts/matmul-roofline-bench.ts` — current kernel vs the tuned-SGEMM ceiling and the bandwidth roofline. This is your scoreboard.
- Add `--dump-wgsl` (supported by the bench harness) to inspect generated WGSL — the fastest way to confirm whether something landed in registers vs spilled to `var ...: array<...>` private memory.
- Always compare against a CPU reference at a small size first; an indexing bug shows up as a large-magnitude mismatch, not a crash.

**Roofline context:** edge-safe compute ceiling ≈ 660 GFLOP/s (aligned, large); streaming bandwidth ≈ 53 GB/s. The compute-heavy layers are firmly **compute-bound** at realistic batch sizes (FLOP:byte ratio grows with M), so optimize the ALU/shared-memory path — not DRAM traffic — until you're near the ceiling. Two realistic caveats the bench now measures: a **ragged `M`** (batch×tokens not a multiple of 64) costs ~10–20% vs the aligned ceiling, and **skinny `N`** (logits) is a different beast entirely (item 4b).
