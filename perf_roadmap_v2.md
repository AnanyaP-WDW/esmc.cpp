# `esmc.cpp` Performance Roadmap v2

Second-generation runtime performance plan. v1 (`perf_roadmap.md`, milestones
M0–M6, experiments EXP-016…EXP-022) is **complete and merged**: weights are
backend-resident, the graph is cached, flash attention is the default, a
`ggml_backend_sched` keeps tensors GPU-resident, `n_threads` is wired to the CPU
backend, and a batched path (`esmc_embed_batch`) reaches 3.6× single-sequence
throughput at batch 16. This document picks up where that left off.

> **Scope.** A *runtime performance* roadmap for the existing custom ggml graph
> (`src/esmc-graph.cpp`, `src/esmc.cpp`). It is not a math change and not a
> rewrite onto the llama.cpp model registry (that remains a decision point, §10).
> **Correctness is a hard gate:** every milestone must hold the M8B/M8C
> cosine-similarity thresholds (`benchmarks/correctness.py`): per-sequence mean
> cosine > 0.999 (F16/Q8_0) / aggregate > 0.995 (Q4_K_*), min cosine > 0.99,
> mean-pool L2 < 0.01. A perf change that moves cosine is a regression, not a win.

---

## 1. Where we are now (v1 exit state)

Steady-state, graph cached, alloc = 0. Apple M4 Max, 36 GB, 300M f16.

| Backend | Bucket | Tokens | compute (ms) | total (ms) | seq/s |
|---------|--------|-------:|-------------:|-----------:|------:|
| Metal | short  | 47  | 22.8 → ~10 (cached) | ~10–11 | 91.7 |
| Metal | medium | 235 | 30.8 → ~27 | ~27 | 36.9 |
| Metal | long   | 850 | 125 → ~172 | ~172 | 5.8 |
| CPU   | short  | 47  | 90.7 | ~91 | ~11 |
| CPU   | medium | 235 | 494 | ~494 | ~2 |
| CPU   | long   | 850 | 2694 | ~2794 | ~0.36 |

Batched (Metal, 28-token seqs): batch 1 → 95 seq/s, batch 4 → 267 (2.8×),
batch 16 → 345 (3.6×), batch 32 → 122–325 (collapses under GPU memory pressure).

Sources: `lab_manual.md` EXP-016 (attribution), EXP-021 (batching), EXP-022
(flash/dense parity), `paper.tex` Table 3 / §5.4.

---

## 2. Root-cause diagnosis v2 (the bottleneck has moved)

v1 removed all per-call *overhead* (alloc/build/upload → 0). The remaining cost
is now **genuine compute and a few structural inefficiencies that only v1's
"repeat the same length" assumption hid.**

| # | Root cause | Evidence | Who it hurts |
|---|------------|----------|--------------|
| V1 | **Activations run in F32 even for the F16 model.** Token embeddings are cast to F32 immediately (`esmc-graph.cpp:138`, `:333`) and every matmul is F16-weight × F32-activation. The M4 Max's native **F16** tensor cores (the reason F16 is the fastest precision, paper §5.2) are therefore only half-used. | `esmc-graph.cpp:138,333`; EXP-022 "FC layers dominate for a 300M model" | All Metal buckets; the recommended config |
| V2 | **Graph cache is keyed by exact `n_tokens`.** A real FASTA with heterogeneous lengths misses the cache on nearly every call, so the graph is rebuilt and re-allocated each time — re-paying exactly the overhead M1 removed. | `esmc-graph.cpp:106` (`cached_n_tokens == n_tokens`), `esmc_reset_compute` | Real batch/FASTA workloads (single-seq path) |
| V3 | **No load-once FASTA path in the CLI.** `esmc-embed` does one sequence per process; embedding a corpus = N process starts × model load (~hundreds of ms each). | `examples/embed/main.cpp` (one `-s` per run) | Real-world corpus embedding (e.g. CLEAN-scale ~220k seqs) |
| V4 | **Batch mask + positions are rebuilt and re-uploaded every call.** The F16 mask `[max_len, max_len, 1, n_seq]` is filled with an O(max_len²·n_seq) host loop and re-uploaded on every `esmc_embed_batch`, even when lengths are unchanged. | `esmc.cpp:632-652` | Batch throughput; worst on long/large batches |
| V5 | **Uniform-length batches still pay for a dense mask.** When all sequences in a batch have equal length (the bucketed case), no mask is needed at all, but a full `[L,L,1,B]` F16 mask is materialized and fed to flash — extra memory traffic and a slower kernel path. | `esmc.cpp:632`, `esmc-graph.cpp:326,371` | Batch ceiling (the batch-32 collapse) |
| V6 | **Two `ggml_scale` nodes per layer for `residue_scale` and one for `1/√head_dim`.** 30 layers × 3 = 90 scale ops that can be folded into the adjacent weights at load time (exact, free). | `esmc-graph.cpp:161,212,223` (and batch `:355,392,403`) | All buckets (small, but free) |
| V7 | **CPU threads = `hardware_concurrency()` (includes E-cores).** On Apple Silicon this oversubscribes the efficiency cores and can *slow* the compute-bound path. | `esmc.cpp:384` | CPU all buckets |
| V8 | **`output.weight` (lm_head) is loaded but never used** by the embedding graph. | `esmc.cpp:107`, loaded into `model->buf` | Load time + RSS (minor) |

**Headline:** V1 (F16 activations) is the biggest single-sequence and batch lever
on the recommended hardware. V2+V3 are the biggest *real-world* throughput levers
(heterogeneous corpora). V4+V5 unlock batch sizes beyond 16.

---

## 3. Guiding principles & guardrails

- **Correctness is a hard gate.** Re-run `benchmarks/correctness.py` at every
  exit; the cosine gates must stay green on every bucket/precision. F16-activation
  work (M-A) is the most correctness-sensitive — gate it hard and keep an F32-
  activation fallback flag.
- **Measure before and after, one variable per milestone.** Use the existing
  `ESMC_PROFILE=1` breakdown + `benchmarks/throughput.py` with identical config.
- **No new dependencies.** Stay within the vendored ggml submodule. (Accelerate
  in M-E is already a ggml CPU option, not a new dep.)
- **Estimated impacts are hypotheses** to validate at each exit, not guarantees.
- **Real-world ≠ microbenchmark.** v2 explicitly adds end-to-end corpus
  throughput (seq/s including model load amortization) as a first-class metric,
  not just steady-state per-forward latency.

---

## 4. Milestone overview (priority order)

| Milestone | Theme | Effort | Risk | Primary beneficiary |
|-----------|-------|--------|------|---------------------|
| M-A | F16 (mixed-precision) activations on Metal | M | **High** (correctness) | All Metal buckets, batch |
| M-B | Length-bucketed graph cache (single-seq) | S–M | Low | Heterogeneous workloads |
| M-C | `--fasta` load-once CLI + length-sorted dynamic batching | M | Low | Real-world corpus throughput |
| M-D | Batch mask/position caching + uniform-length fast path | M | Med | Batch throughput + ceiling |
| M-E | CPU path: P-core threads, Accelerate/AMX, flash-on-CPU | M | Low | CPU/portability story |
| M-F | Graph hygiene: fold scales into weights, drop dead lm_head | S | Low | All buckets (small, free) |
| M-G | Exploratory: quantized Metal matmul / repacking | M | Med | Memory-bound, low ROI |
| M-H | Scaling & concurrency: 600M/6B + multi-stream; arch-port decision | XL | High | Long-term |

Effort: S ≤ 1 day, M ≈ 2–4 days, L ≈ 1–2 weeks, XL ≈ multi-week.

**Dependency order:** M-F (free, do first) → M-A → M-B → (M-C, M-D parallel) →
M-E → M-G → M-H. M-A should land early because every later throughput number
depends on whether activations are F16 or F32.

---

## 5. Milestones in detail

### M-F — Graph hygiene (do first; free and exact)

**Goal.** Remove provably-redundant graph nodes so later measurements aren't
polluted by trivially-removable ops.

**Changes.**
- **Fold `residue_scale` into weights at load.** `cur = wo·attn / residue_scale`
  and `cur = ffn_down·x / residue_scale` (`esmc-graph.cpp:211-212`, `:222-223`).
  Pre-multiply `layer.wo` and `layer.ffn_down` by `1/residue_scale` once in
  `esmc_load_model` (after the backend copy). Removes 60 `ggml_scale` nodes.
- **Fold `1/√head_dim` into `wq`.** Pre-scale `layer.wq` at load so the per-layer
  `ggml_scale(Q, …)` (`:161`, `:355`) disappears. (Keep the explicit scale only on
  the debug `--check-layer0-qk` path, which reads Q pre-attention.)
- **Drop the dead `output.weight` load** (V8) when no scoring API is active:
  skip it in `esmc_load_tensors`/the `model->buf` copy. Saves load time + RSS.

**Caveat.** Folding into *quantized* weights must happen on the dequantized
values or be applied as a separate F32 scale vector — pre-multiplying a Q4_K block
in place is not exact. For quantized models, fold only where the tensor is F16/F32,
else keep the scale node. Gate per-tensor on `ggml_is_quantized(t->type)`.

**Estimated impact.** 1–4% on short/medium (fewer nodes, less kernel launch);
mainly it cleans the graph for M-A.

**Effort / Risk.** S / Low.

**Exit criteria.**
- [ ] Graph node count drops by ≥ 90 nodes for the 30-layer model (log
      `ggml_graph_n_nodes` before/after).
- [ ] `benchmarks/correctness.py` cosine gates green on **all** precisions
      (F16/Q8_0/Q4_K_M/Q4_K_S) — folding must not perturb quantized paths.
- [ ] Steady-state latency not worse than v1 baseline on any bucket.

---

### M-A — F16 (mixed-precision) activations on Metal

The single biggest lever for the recommended config. The model weights are F16,
the M4 Max has native F16 tensor cores, and EXP-022 found the fully-connected
layers (not attention) dominate at 300M — yet activations are upcast to F32 at the
embedding lookup and stay F32 through every matmul.

**Change.** Keep the activation stream in F16 through the matmul-heavy path,
upcasting to F32 only where numerically required (LayerNorm reductions, softmax,
RoPE if needed, final output).
- Remove/relocate the `ggml_cast(cur, GGML_TYPE_F32)` after `ggml_get_rows`
  (`esmc-graph.cpp:138`, `:333`); let the embedding stay F16.
- Keep `ggml_norm` (LayerNorm) computing in F32 internally — ggml's norm
  reduces in F32 regardless of input type; feed it F16, take F16 out, then matmul
  in F16.
- Ensure flash attention runs F16 in / F16 out (it already accumulates F32
  internally per EXP-022).
- Cast to F32 once at the very end before readback (already done, `:228`/`:408`).
- Gate the whole thing behind a runtime flag (`esmc_context_set_f16_activations`,
  default off until the cosine gate proves it), mirroring the existing
  `use_flash_attn` flag plumbing (`esmc.cpp:409`).

**Risk.** This is where silent numerical error lives (cf. the paper's whole
thesis). LayerNorm in F16, accumulation of 30 residual adds in F16, and the
SwiGLU product are the danger spots. Mitigation: keep all *norm* and *residual
add* accumulations in F32; only the large GEMM inputs go F16. This is standard
mixed precision, but must be proven against the 100-sequence reference, not a
smoke test (BUG-002 lesson).

**Estimated impact.** Hypothesis: 1.3–1.8× on medium/long Metal (matmul-bound) and
a meaningful batch-throughput lift, by actually using the F16 tensor-core path.
Short bucket may see less (launch-bound). **Validate, don't assume.**

**Effort / Risk.** M / **High**.

**Exit criteria.**
- [ ] `benchmarks/correctness.py` with F16 activations ON: F16 model still
      mean cosine > 0.999, min cosine > 0.99, mean-pool L2 < 0.01 on all 100
      sequences (must include N/Q-containing and long sequences).
- [ ] CPU/Metal parity check still > 0.9999 (`tests/validate.py --compare-cpu-metal`).
- [ ] Re-run `throughput.py`: **Metal medium ≥ 1.25× v1 medium seq/s** (≥ ~46 seq/s)
      OR a documented finding that F16 activations do not help on this kernel set
      (a valid negative result — record it and keep the flag off by default).
- [ ] Default-off flag verified: with the flag off, output is bit-identical to v1.

---

### M-B — Length-bucketed graph cache (single-sequence path)

**Goal.** Make the cached-graph win survive heterogeneous-length workloads (V2).

**Change.**
- Replace the single `cached_n_tokens`/`cached_gf` slot with either:
  (a) **bucketed padding** — round `n_tokens` up to a small set of buckets
  (e.g. 64/128/256/512/1024/2048), build one graph per bucket, pad inputs and mask
  the pad positions; or (b) a **small LRU** of `(n_tokens → graph)` entries
  (e.g. 8 slots) keyed exactly.
- Bucketing (a) is preferred: it bounds the number of distinct graphs and reuses
  the batch path's masking machinery. Pad with `pad_id`, attend only to real
  tokens (reuse the M-D mask), and slice the real rows on readback.
- Touchpoints: `esmc-internal.h:23-30` (cache fields), `esmc_build_graph`
  (`:105-108`), `esmc_run_graph` readback in `esmc.cpp:550-574`.

**Estimated impact.** On a real FASTA (all-distinct lengths), v1 silently rebuilds
every call; bucketing restores alloc≈0 and build≈0 for ~99% of calls. Predicted
2–5× on heterogeneous single-seq streams (recovers the M1 win that the exact-match
key throws away). No change to uniform-length microbenchmarks.

**Effort / Risk.** S–M / Low.

**Exit criteria.**
- [ ] On a 100-sequence all-distinct-length FASTA, graph is built ≤ (number of
      buckets) times, not 100 times (assert/log rebuild count).
- [ ] Padded embeddings bit-match unpadded within F16 noise (≤ 5e-4) per sequence.
- [ ] End-to-end seq/s on that FASTA ≥ 2× the v1 exact-key path (in-process loop).
- [ ] Padding waste < 30% on a realistic Swiss-Prot length distribution.

---

### M-C — `--fasta` load-once CLI + length-sorted dynamic batching

**Goal.** Real-world corpus throughput (V3) — the metric that matters for
downstream users embedding thousands of sequences (e.g. a CLEAN-style EC pipeline).
Today the model is reloaded per process; this milestone amortizes load over the
whole corpus and feeds the batch path intelligently.

**Change.**
- Add `--fasta IN.fasta --output-dir DIR/` (or a single stacked `.npy`) to
  `esmc-embed`: parse FASTA, load the model once, iterate.
- **Length-sort** the corpus, then form batches by a **token budget**
  (`n_seq × max_len ≤ B_tok`) rather than a fixed `n_seq`. This both minimizes
  padding (similar lengths batched together) and avoids the batch-32 memory
  collapse (the budget caps GPU attention scratch).
- Reuse `esmc_embed_batch`; choose `max_len` per batch from the longest member.
- Emit per-sequence outputs in input order (carry an index permutation).
- Optional `--pool mean` to write one vector per sequence (pairs with M-I below).

**Estimated impact.** For a corpus, end-to-end throughput goes from
"model-load-bound" to "compute-bound." Combined with length-sorting, expect
near the batch-8–16 steady-state rate (≈ 290–345 seq/s on short/medium) sustained
across the whole file, versus a handful of seq/s when each sequence pays a model
load. This is the headline real-world number.

**Effort / Risk.** M / Low.

**Exit criteria.**
- [ ] `esmc-embed --fasta benchmarks/sequences_throughput.fasta --output-dir /tmp/out`
      loads the model exactly once (verify via a single "weight buffer allocated" log).
- [ ] Aggregate seq/s on a ≥ 1000-sequence FASTA ≥ 10× a `for seq in fasta: esmc-embed -s`
      shell loop (model-reload baseline).
- [ ] Per-sequence outputs identical (within F16 noise) to single-sequence runs,
      in correct input order.
- [ ] Token-budget batching keeps peak RSS within the 36 GB budget at the largest
      configured budget (no batch-32-style collapse).

---

### M-D — Batch mask/position caching + uniform-length fast path

**Goal.** Recover batch throughput lost to host-side mask work (V4) and raise the
batch ceiling (V5).

**Change.**
- **Cache the position tensor** and only re-upload when `max_len` changes
  (`esmc.cpp:622-630`).
- **Cache the mask** keyed by `(max_len, lengths-vector)`; skip the O(max_len²·n_seq)
  rebuild + re-upload when the length signature is unchanged (`esmc.cpp:632-652`).
- **Uniform-length fast path:** when all `lengths[s] == max_len` (no padding —
  the M-C length-sorted batches will frequently hit this), pass `mask = nullptr`
  to `ggml_flash_attn_ext` (encoder is non-causal; no mask needed). This skips the
  mask materialization, the upload, and uses the faster maskless flash kernel.
  Touchpoints: `esmc-graph.cpp:326,371` (make the mask input optional) and
  `esmc.cpp:632`.

**Estimated impact.** Removes a per-call host loop that grows quadratically with
length; on long/large batches this is a real fraction of wall time. The maskless
path should also lift the batch-32 ceiling by cutting attention-scratch memory.
Predicted: batch-16 throughput +10–25%; usable batch sizes extended past 16.

**Effort / Risk.** M / Medium (mask correctness — guard with `esmc-test-batch`).

**Exit criteria.**
- [ ] On repeated equal-shape batches, mask/position upload bytes after the first
      call = 0 (verify via profiling counter).
- [ ] `esmc-test-batch` passes (batched vs single ≤ 4.2e-4) for both the masked
      (ragged) and maskless (uniform) paths.
- [ ] Batch-16 seq/s ≥ 1.1× v1 batch-16 (345 → ≥ 380).
- [ ] At least one batch size > 16 sustains > 345 seq/s without the variance
      collapse (uniform-length, maskless).

---

### M-E — CPU path: P-core threads, Accelerate/AMX, flash-on-CPU

**Goal.** Make the portability story (no-GPU hosts) competitive, since the CPU
path is 10–40× slower than Metal and quantization did **not** help it (EXP-012).

**Change.**
- **Thread count (V7):** default to *performance* core count on Apple Silicon, not
  `hardware_concurrency()` (which includes E-cores). Detect via
  `sysctl hw.perflevel0.physicalcpu`; expose `esmc_set_n_threads`. Benchmark the
  sweet spot; oversubscription of E-cores can regress compute-bound work.
- **Accelerate / AMX:** confirm the ggml CPU backend is built with the Apple
  Accelerate path (`GGML_ACCELERATE`/`GGML_BLAS`) for the large GEMMs; measure F16
  vs F32 GEMM on CPU.
- **Flash-on-CPU:** measure `ggml_flash_attn_ext` vs the dense path on CPU for the
  long bucket; pick the winner per-backend (flash is default-on now but was tuned
  on Metal).
- **Quant re-evaluation:** with correct threading, re-test whether Q8_0/Q4_K_M CPU
  is now bandwidth-bound enough to beat F16 CPU (EXP-012 said no at 4 threads).

**Estimated impact.** CPU medium/long 1.3–1.8× from correct core selection alone;
Accelerate may add more on the GEMM-bound long bucket.

**Effort / Risk.** M / Low.

**Exit criteria.**
- [ ] CPU backend reports the chosen (P-core) thread count; override works.
- [ ] `throughput.py` CPU medium ≥ 1.3× v1 CPU medium seq/s.
- [ ] Documented per-backend flash vs dense decision in `lab_manual.md`.
- [ ] Cosine gates green (threading/Accelerate must not change math).

---

### M-G — Exploratory: quantized Metal matmul / repacking

**Goal.** Revisit whether quantized weights can be made *faster* (not just
smaller) on Metal, given the paper's finding that dequant overhead currently
negates bandwidth savings at 300M.

**Change (investigate, low commitment).**
- Profile the Metal quantized matmul kernels (`mul_mv` vs `mul_mm`) actually
  selected for the ESM-C shapes; check whether the 960-wide (non-256-multiple)
  matrices fall onto a slow path.
- Evaluate weight repacking / `Q4_0` → `Q4_0_4_4`-style layouts if available in
  the vendored ggml for the relevant backend.
- This is bandwidth-vs-compute; document the regime where (if ever) quant wins on
  throughput, otherwise close it out as "size-only," confirming the paper.

**Estimated impact.** Likely small at 300M; this milestone's value is a
definitive measured answer (and groundwork for 6B in M-H, where memory bandwidth
binds).

**Effort / Risk.** M / Medium. **Lowest priority** — skip if M-A/M-C deliver.

**Exit criteria.**
- [ ] A measured table of Metal quant-matmul kernel selection + timing for ESM-C
      shapes in `lab_manual.md`.
- [ ] Either a throughput win (Q* Metal ≥ 1.0× F16 Metal on some bucket) or a
      documented confirmation that quant is size-only on this hardware.

---

### M-H — Scaling & concurrency; arch-port decision

**Goal.** Carry forward v1's M6 decision and address the paper's explicit future
work: 600M/6B bring-up, multi-hardware, and concurrent streams.

**Change.**
- **Multi-stream / multi-context:** allow N `esmc_context`s sharing one
  `esmc_model` (weights are already a single resident buffer) to overlap host prep
  with GPU compute, or to serve concurrent requests. Verify thread-safety of the
  shared `model->buf` (read-only) vs per-context `sched`.
- **600M/6B:** at 6B, `d_model = 2560` (a 256 multiple) → real k-quant throughout
  and a *memory-bound* regime, which flips the M-G calculus and raises the value of
  quant + mmap. Scope a bring-up sub-plan.
- **Arch-port decision (from v1 M6):** with M-A…M-E measured, decide go/no-go on
  porting ESM-C to a first-class llama.cpp architecture to inherit kernels for
  free. Honor `ggml/AGENTS.md` AI-contribution policy (fork, not upstream).

**Effort / Risk.** XL / High.

**Exit criteria (decision, not full port).**
- [ ] Multi-context concurrency benchmark: aggregate seq/s with 2–4 contexts vs 1.
- [ ] Written go/no-go on the arch port in `lab_manual.md`, citing M-A…M-E ratios.
- [ ] If go: a scoped sub-plan with its own milestones and exit gates.

---

## 6. Optional adjacent work (not throughput, but requested by users)

### M-I — On-device mean pooling (`esmc_embed_mean`)

`esmc_embed_mean` is currently a stub (`esmc.cpp:719`). For the dominant
"one vector per sequence" use case (variant effect, CLEAN-style EC annotation),
compute the CLS/EOS-stripped mean **inside the graph** (`ggml_mean` / `ggml_sum_rows`
over the residue axis; both exist in the vendored `ggml.h:1036,1045`) and read
back `[n_embd]` instead of `[n_embd × n_tokens]`.

Note: readback is already < 0.3 ms (EXP-016), so this is **not** a speed
milestone — its value is a clean API, lower host memory, and less host-side
pooling code in `examples/embed/main.cpp:54-72`. Prioritize only if a downstream
consumer needs it.

**Exit criteria.**
- [ ] `esmc_embed_mean` returns the same vector (≤ 1e-5) as host-side mean of the
      per-residue path.
- [ ] Readback size for mean mode = `n_embd` floats.

---

## 7. Cumulative targets (hypotheses to validate)

Single-sequence, Metal, 300M f16, steady state (seq/s). v1 = current.

| Stage | short | medium | long |
|-------|------:|-------:|-----:|
| v1 (current) | 91.7 | 36.9 | 5.8 |
| + M-F | ~93 | ~38 | ~5.9 |
| + M-A (if F16 acts help) | ~100 | ~48 | ~8 |
| + M-D (batch-16 aggregate) | — | ≥ 380 seq/s | — |

Real-world corpus (heterogeneous FASTA, end-to-end incl. load), the headline v2 win:

| Stage | ~1000-seq FASTA aggregate seq/s |
|-------|--------------------------------:|
| Shell loop (`esmc-embed -s` per seq, reload) | baseline (≈ a few seq/s) |
| + M-B (bucketed cache, single-seq loop) | ≥ 2× baseline |
| + M-C (load-once + dynamic batching) | ≥ 10× baseline (toward 290–345 seq/s sustained) |

---

## 8. Measurement protocol (applied at every exit)

1. **Correctness first:** `.venv/bin/python benchmarks/correctness.py` — cosine
   gates must match v1 (no regression on any bucket/precision). For M-A, this is
   the gate, run with the flag both off and on.
2. **Attribution:** capture `ESMC_PROFILE=1` build/alloc/compute/readback for
   short/medium/long (and per-batch for batch milestones).
3. **Throughput:** rebuild and run the existing harness with identical config:
   ```bash
   cmake --build build --target esmc-bench
   RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
   .venv/bin/python benchmarks/throughput.py \
     --config benchmarks/config_throughput_300m.json \
     --iterations 1000 \
     --output-prefix "results/throughput_${RUN_ID}" \
     2>&1 | tee "results/logs/throughput_${RUN_ID}.log"
   ```
4. **Real-world (new in v2):** time `esmc-embed --fasta` over a fixed ≥ 1000-seq
   FASTA, report aggregate seq/s including model load, vs the shell-loop baseline.
5. **Record:** append a dated `EXP-0xx` row to `lab_manual.md` with before/after
   numbers and the exit-criteria checklist outcome.

Hold every variable constant except the milestone under test: same host, same
model file, same sequences (`benchmarks/sequences_throughput.fasta`), same
warmup/iteration counts.

---

## 9. Risks, dependencies, rollback

| Risk | Mitigation |
|------|------------|
| **F16 activations (M-A) move cosine past tolerance** | Default-off runtime flag; keep F32 norm/residual accumulation; gate on the full 100-seq reference incl. N/Q + long sequences (BUG-002 lesson). Negative result is acceptable and recorded. |
| Scale-folding (M-F) corrupts quantized weights | Fold only into F16/F32 tensors; keep the scale node when `ggml_is_quantized(t)`. Verify all 4 precisions. |
| Bucketed padding (M-B) introduces pad-attention leakage | Reuse the validated batch mask; assert padded == unpadded within F16 noise. |
| Maskless uniform path (M-D) silently wrong on ragged input | Only enable when `all(lengths == max_len)`; `esmc-test-batch` covers both paths. |
| Dynamic batching (M-C) OOMs on a pathological long sequence | Token-budget cap; fall back to single-sequence when one sequence exceeds the budget. |
| CPU thread change (M-E) regresses on non-Apple hosts | Detect platform; default sensibly; allow override; benchmark on target. |

**Dependency order:** M-F → M-A → M-B → (M-C ∥ M-D) → M-E → M-G → M-H.
M-A first among substantive items because all later throughput numbers depend on
the activation precision in force.

---

## 10. Out of scope

- Algorithmic changes to the ESM-C math (architecture is fixed and validated).
- Training / fine-tuning paths (no backward graph in this runtime).
- Multi-GPU / distributed inference.
- Upstreaming to `ggml-org/llama.cpp` (see `ggml/AGENTS.md` AI-contribution policy);
  an arch port (M-H) would live in this repo's fork.

---

## 11. Paper impact (if v2 milestones land)

If M-A and/or M-C produce material gains, `paper.tex` §5.2 (throughput), §5.3
(perf roadmap), §5.4 (batching), and the abstract should be revised — most
naturally by adding a "v2" paragraph to §5.3 and a real-world corpus-throughput
number, and softening the "batch 16 is the sweet spot" claim in §6 if M-D extends
the ceiling. Follow the same task/exit-criterion structure as `perf_roadmap.md`
§10 (T1–T13). Do this only after the measured numbers exist — never edit the paper
ahead of the data.
