# `esmc.cpp` Performance Roadmap

Prioritized engineering plan to close the throughput gap with PyTorch (CPU and MPS),
grounded in the current `esmc.cpp` runtime structure. Each milestone has a concrete
change set, the exact code touchpoints, an estimated impact, effort/risk, and a
measurable exit criterion.

> **Scope.** This is a *runtime performance* roadmap for the existing custom ggml graph
> (`src/esmc-graph.cpp`). It is **not** a rewrite onto the llama.cpp model registry; that
> option is captured as a strategic decision point in §9. No correctness behavior may
> change: every milestone must hold the M8B/M8C cosine-similarity gates.

---

## 1. Baseline (Milestone 8E, 300M f16, `results/throughput_*_20260530_104536.csv`)

Measured on Apple Silicon, 1000 iterations, warmup 10, single sequence per bucket.

| Bucket | tokens | esmc CPU | esmc Metal | PyTorch CPU (f32) | PyTorch MPS (f32) |
|--------|-------:|---------:|-----------:|------------------:|------------------:|
| short  | 47  | 165.1 ms / 5.79 seq/s | 114.4 ms / 8.76 seq/s | 96.1 ms / 10.29 seq/s | 33.6 ms / 29.33 seq/s |
| medium | 235 | 609.0 ms / 1.65 seq/s | 211.3 ms / 4.73 seq/s | 218.6 ms / 4.61 seq/s | 98.1 ms / 10.10 seq/s |
| long   | 850 | 2933.4 ms / 0.33 seq/s | 774.5 ms / 1.26 seq/s | 571.0 ms / 1.75 seq/s | 350.9 ms / 2.83 seq/s |

### Relative to PyTorch CPU (seq/s ratio)

| Bucket | esmc CPU / PT CPU | esmc Metal / PT CPU | esmc Metal / PT MPS |
|--------|------------------:|--------------------:|--------------------:|
| short  | 0.56× | 0.85× | 0.30× |
| medium | 0.36× | 1.03× | 0.47× |
| long   | 0.19× | 0.72× | 0.45× |

**Reading the data.**

1. The CPU gap **widens with length** (0.56× → 0.36× → 0.19×): a signature of two
   compounding problems — fixed per-call overhead that dominates short sequences, and
   naive O(n²) attention that dominates long sequences.
2. Metal already **matches PyTorch CPU at medium length** (1.03×), so the ggml/Metal
   kernels are sound. The remaining gap to MPS is fused attention + GPU residency.

---

## 2. Root-cause diagnosis (current code)

| # | Root cause | Evidence (file:line) | Who it hurts |
|---|------------|----------------------|--------------|
| R1 | Graph + compute buffer destroyed and rebuilt every forward | `esmc_build_graph` → `esmc_reset_compute` (`src/esmc-graph.cpp:91`, `:233`) | All buckets; dominates short |
| R2 | All weights re-uploaded to backend every forward (~634 MiB f16) | `esmc_alloc_compute` (`src/esmc-graph.cpp:48-60`) | All buckets; severe on Metal (host→device) |
| R3 | `n_threads` stored but never applied to CPU backend | `esmc-internal.h:24`, `src/esmc.cpp:277,287`; no `set_n_threads` call before `ggml_backend_graph_compute` (`src/esmc-graph.cpp:227`) | CPU, all buckets |
| R4 | Dense attention, no flash; forced F32 matmul precision | `src/esmc-graph.cpp:170-177` (`ggml_mul_mat_set_prec(..., GGML_PREC_F32)`) | Long bucket especially |
| R5 | Redundant `ggml_cont`/`permute` materializations around attention | `src/esmc-graph.cpp:165-177` | Medium/long |
| R6 | Weights never resident on backend at load; `model->buf` unused | `esmc_load_model` (`src/esmc.cpp:209-249`) | Enables R2 fix |
| R7 | No batching / multi-sequence packing | whole runtime is single-sequence | Throughput-bound workloads |

**Headline:** R1 + R2 are pure overhead paid on *every* call and are the cheapest to fix.
R4 is the structural fix for the long-sequence collapse.

---

## 3. Guiding principles & guardrails

- **Correctness is a hard gate.** Every milestone re-runs `benchmarks/correctness.py`
  and must keep the M8C cosine thresholds (per-bucket) green. A perf change that moves
  cosine is a regression, not a win.
- **Measure before and after.** Use the existing harness (`benchmarks/throughput.py`)
  with identical config; never eyeball.
- **One variable per milestone.** Land and benchmark each item independently so impact
  is attributable.
- **No new dependencies.** Stay within the vendored ggml/llama.cpp submodule.
- **Estimated impacts are hypotheses.** Ranges below are predictions to validate at each
  milestone's exit, not guarantees.

---

## 4. Milestone overview (priority order)

| Milestone | Theme | Effort | Risk | Primary beneficiary |
|-----------|-------|--------|------|---------------------|
| M0 | Instrumentation & attribution | S | Low | All |
| M1 | Kill per-call overhead (graph + weight residency + threads) | M | Low | CPU all, Metal all |
| M2 | Efficient attention (flash + precision + fewer copies) | M | Med | Long bucket |
| M3 | Metal/GPU residency toward MPS parity | M | Med | Metal all |
| M4 | Quantized throughput (Q8_0 / Q4_K_M) | S–M | Low | CPU, memory-bound |
| M5 | Batching & bucketed padding | L | Med | Throughput workloads |
| M6 | Strategic decision: llama.cpp arch port | XL | High | Long-term maintenance |

Effort: S ≤ 1 day, M ≈ 2–4 days, L ≈ 1–2 weeks, XL ≈ multi-week.

---

## 5. Milestones in detail

### M0 — Instrumentation & attribution

**Goal.** Replace guesswork with a per-stage time breakdown so every later milestone can
prove its impact and we know whether overhead (R1/R2) or compute (R4) dominates each
bucket.

**Changes.**
- Add coarse timers inside `esmc_embed` around: graph build, weight upload/alloc, compute,
  and output readback (`src/esmc.cpp:417-456`, `src/esmc-graph.cpp`).
- Expose `GGML_OPENVINO_PROFILING`-style opt-in via an env var (e.g. `ESMC_PROFILE=1`)
  that prints the four sub-timings; default off so the benchmark path is unaffected.
- Optionally enable `ggml_graph_print` / per-node timing under the same flag.

**Estimated impact.** 0× direct, but unblocks correct attribution for M1–M4.

**Effort / Risk.** S / Low.

**Exit criteria.**
- [ ] `ESMC_PROFILE=1 ./build/esmc-bench ...` prints build/upload/compute/readback ms for
      short, medium, long.
- [ ] Breakdown recorded in `lab_manual.md` as the new baseline attribution table.
- [ ] Profiling adds < 1% overhead when disabled (verified against current numbers).

---

### M1 — Eliminate per-call overhead

Single highest-ROI milestone. Targets R1, R2, R3, R6. Low risk because none of it changes
the math.

#### M1.1 — Upload weights to the backend once

**Change.** At model load (or first context creation), import each weight tensor into a
persistent backend buffer (`model->buf`) and `ggml_backend_tensor_set` **once**. The
compute graph then references the already-resident weights instead of re-copying them.
- Touchpoints: `esmc_load_model` (`src/esmc.cpp:209-249`), `esmc_alloc_compute`
  (`src/esmc-graph.cpp:47-63`), `esmc_import_tensor` (`src/esmc-graph.cpp:11-27`).
- Remove the per-build `ggml_backend_tensor_set` loop from the compute path.

**Estimated impact.** Removes a ~634 MiB copy per forward. Largest relative win on short
CPU and on all Metal buckets (host→device upload eliminated). Predicted: short CPU
−20–40% latency, Metal all −10–30%.

#### M1.2 — Cache and reuse the compute graph per length bucket

**Change.** Stop calling `esmc_reset_compute` on every forward. Build the graph once per
distinct `n_tokens` (or per padded bucket), keep `ctx_compute` + `buf_compute` alive on
the `esmc_context`, and only rebuild when the token count changes. Use a graph allocator
(`ggml_gallocr`) so input tensors can be reset and the graph recomputed.
- Touchpoints: `esmc_build_graph` (`src/esmc-graph.cpp:86-204`), `esmc_run_graph`
  (`:207-231`), `esmc_reset_compute` (`:233-246`), `esmc_context` struct
  (`src/esmc-internal.h:13-26`).
- Add a small cache keyed by `n_tokens` (or a fixed bucket set) on `esmc_context`.

**Estimated impact.** Removes graph construction + buffer alloc/free per call. Predicted:
short/medium CPU −15–30% latency; compounds with M1.1.

#### M1.3 — Wire `n_threads` to the CPU backend

**Change.** Before `ggml_backend_graph_compute`, call the CPU backend's
`ggml_backend_cpu_set_n_threads` (resolved via `ggml_backend_reg_get_proc_address`,
mirroring `clip.cpp:4114-4116`). Default `n_threads` to physical core count instead of the
hardcoded 4 (`src/esmc.cpp:277,287`).
- Touchpoints: `esmc_run_graph` (`src/esmc-graph.cpp:227`), context init
  (`src/esmc.cpp:281-289`).

**Estimated impact.** CPU only. On an 8+ core machine, going from 4 → physical cores can
yield 1.3–1.8× on compute-bound (medium/long) buckets.

**M1 effort / risk.** M / Low.

**M1 exit criteria.**
- [ ] Per-forward weight upload bytes = 0 after first call (confirmed via M0 profiling).
- [ ] Graph rebuilt only when `n_tokens` changes (assert/log on rebuild).
- [ ] CPU backend reports the configured thread count.
- [ ] `benchmarks/correctness.py` cosine gates unchanged (bit-comparable to baseline).
- [ ] Re-run `throughput.py`: **esmc CPU short ≥ 0.85× PT CPU** and **esmc CPU medium ≥ 0.6× PT CPU** (up from 0.56× / 0.36×).

---

### M2 — Efficient attention

Targets R4 and R5 — the structural cause of the long-sequence collapse (0.19× CPU).

#### M2.1 — Flash attention

**Change.** Replace the explicit `KQ → soft_max → KQV` sequence with
`ggml_flash_attn_ext` on backends that support it (Metal first, then CPU). Keep the
current dense path as a correctness fallback behind a flag.
- Touchpoints: `src/esmc-graph.cpp:160-177`. Reference usage:
  `ggml/src/llama-graph.cpp:1976-1998` (`use_flash_attn`, `ggml_flash_attn_ext`,
  `ggml_flash_attn_ext_set_prec`).
- Note: ESM-C attention is **non-causal** (encoder), so the mask is all-visible; pass a
  null/zero mask and the `1/sqrt(head_dim)` scale already carried by Q.

**Estimated impact.** Long bucket is attention-dominated. Predicted Metal long
0.45× → 0.7–0.9× of PT MPS; CPU long 0.19× → 0.4–0.6× of PT CPU.

#### M2.2 — Drop forced F32 precision where safe + reduce copies

**Change.** Re-evaluate the `ggml_mul_mat_set_prec(KQ/KQV, GGML_PREC_F32)` calls
(`src/esmc-graph.cpp:171,175`). If flash attention internally uses F32 accumulation and
the cosine gates hold, remove the explicit F32 forcing. Audit the three
`ggml_cont(ggml_permute(...))` materializations (`:165-177`) and eliminate any that flash
attention makes unnecessary.

**Estimated impact.** Medium/long −5–15% latency; frees memory bandwidth.

**M2 effort / risk.** M / Medium (correctness-sensitive — guard with the cosine gates).

**M2 exit criteria.**
- [ ] Flash path active on Metal and CPU; dense fallback selectable via flag.
- [ ] `benchmarks/correctness.py` cosine gates green on all buckets with flash enabled.
- [ ] Re-run `throughput.py`: **esmc Metal long ≥ 0.7× PT MPS** (from 0.45×) and
      **esmc CPU long ≥ 0.4× PT CPU** (from 0.19×).
- [ ] Long-bucket scaling curve (latency vs tokens) is sub-quadratic vs baseline.

---

### M3 — Metal/GPU residency toward MPS parity

**Goal.** Close the remaining Metal-vs-MPS gap by keeping everything GPU-resident and
minimizing host↔device traffic.

**Changes.**
- Introduce `ggml_backend_sched` so weights, KV/intermediates, and the graph live on the
  Metal device across calls (reference: `clip.cpp:4120`, `llama-graph.cpp` sched usage).
- Keep input token IDs / positions as the only per-call host→device upload; read back only
  the final embedding tensor.
- Confirm `ggml_cast` to F32 at input/output (`src/esmc-graph.cpp:118,196`) is not forcing
  redundant device round-trips.

**Estimated impact.** Metal all buckets toward 0.7–1.0× of PT MPS for medium/long.

**Effort / risk.** M / Medium.

**Exit criteria.**
- [ ] Per-call host↔device bytes = token inputs + final embedding only (verified).
- [ ] Cosine gates green.
- [ ] **esmc Metal medium ≥ 0.8× PT MPS** and **esmc Metal long ≥ 0.8× PT MPS**.

---

### M4 — Quantized throughput (the real value proposition)

**Goal.** Demonstrate the regime where `esmc.cpp` should clearly beat PyTorch: low-memory,
quantized CPU inference. Quant artifacts already exist (Q8_0 337 MiB, Q4_K_M 237 MiB per
`lab_manual.md`).

**Changes.**
- Add Q8_0 and Q4_K_M to the throughput matrix in `benchmarks/throughput.py` /
  `benchmarks/config_throughput_300m.json`.
- Verify the ggml quantized matmul paths are exercised by the ESM-C graph (no F32
  upcasting of weights in `esmc_import_tensor`).
- Confirm M8C cosine thresholds for quantized precisions still hold under the new perf
  code (M1–M3 must not have changed quant behavior).

**Estimated impact.** CPU latency −20–40% vs f16 from reduced memory bandwidth; this is
where `esmc.cpp` can exceed PyTorch CPU outright.

**Effort / risk.** S–M / Low.

**Exit criteria.**
- [ ] Q8_0 and Q4_K_M rows present in `results/throughput_*.csv`.
- [ ] Q8_0 cosine gate green; Q4_K_M within documented M8C tolerance.
- [ ] **esmc CPU (Q4_K_M) medium ≥ 1.0× PT CPU (f32)** — i.e. a bucket where we win.

---

### M5 — Batching & bucketed padding

**Goal.** Optimize for throughput-bound workloads (FASTA batch embed, ProteinGym
sweeps) rather than single-sequence latency.

**Changes.**
- Extend the graph and API to accept a padded batch `[n_seq, max_len]` with an attention
  mask, instead of one sequence per call.
- Length-bucket sequences to bound padding waste; reuse the M1.2 per-bucket cached graph.
- Add a batched path to `esmc_embed` and a `--fasta` batch mode in the bench/embed CLIs.

**Estimated impact.** 1.5–4× aggregate seq/s on multi-sequence workloads via better core/
GPU utilization and amortized overhead. Does not change single-sequence latency.

**Effort / risk.** L / Medium (graph + API surface change; mask correctness).

**Exit criteria.**
- [ ] Batched embeddings bit-match per-sequence embeddings within cosine gate.
- [ ] Aggregate seq/s on a 100-sequence FASTA ≥ 1.5× the per-call loop at batch size ≥ 8.
- [ ] Padding overhead < 20% on a realistic length distribution.

---

### M6 — Strategic decision point: llama.cpp arch registration

**Goal.** Decide whether to port ESM-C to a first-class llama.cpp architecture (like
`bert.cpp`, `nomic-bert.cpp`, `llama-embed.cpp`) to inherit graph reuse, flash attention,
quant improvements, backend scheduling, and new backends "for free" long-term.

**Decision inputs (gathered from M1–M5 results).**
- If M1–M4 already reach ≥ 0.8× PT MPS and ≥ 1.0× PT CPU (quant), the custom graph is
  good enough — defer the port.
- If maintenance burden of the custom graph grows (new backends, flash-attn API churn),
  the port pays off.

**Note.** A port is **not** "use llama.cpp instead of ggml" — llama.cpp *is* ggml plus the
inference infrastructure. The vendored submodule's `ggml/AGENTS.md` also restricts
AI-generated upstream PRs; an arch port would live in this repo's fork, not upstream.

**Effort / risk.** XL / High.

**Exit criteria (for the decision, not the port).**
- [ ] Written go/no-go in `lab_manual.md` citing M1–M5 measured ratios.
- [ ] If go: a scoped sub-plan with its own milestones; if no-go: documented rationale.

---

## 6. Cumulative targets

Predicted trajectory of the seq/s ratios (validate at each exit; these are hypotheses).

| Stage | short CPU/PT CPU | medium CPU/PT CPU | long CPU/PT CPU | medium Metal/PT MPS | long Metal/PT MPS |
|-------|-----------------:|------------------:|----------------:|--------------------:|------------------:|
| Baseline | 0.56× | 0.36× | 0.19× | 0.47× | 0.45× |
| After M1 | ~0.9× | ~0.7× | ~0.4× | ~0.6× | ~0.55× |
| After M2 | ~0.9× | ~0.8× | ~0.6× | ~0.75× | ~0.8× |
| After M3 | ~0.9× | ~0.8× | ~0.6× | ~0.85× | ~0.85× |
| After M4 (Q4_K_M) | ≥1.0× | ≥1.0× | ~0.7× | n/a | n/a |

---

## 7. Measurement protocol (applied at every exit)

1. **Correctness first:** `.venv/bin/python benchmarks/correctness.py` — cosine gates must
   match baseline (no regression on any bucket/precision).
2. **Throughput:** rebuild `esmc-bench`, then
   ```bash
   cmake --build build --target esmc-bench
   RUN_ID="$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
   .venv/bin/python benchmarks/throughput.py \
     --config benchmarks/config_throughput_300m.json \
     --iterations 1000 \
     --output-prefix "results/throughput_${RUN_ID}" \
     2>&1 | tee "results/logs/throughput_${RUN_ID}.log"
   ```
3. **Attribution:** capture the `ESMC_PROFILE=1` breakdown (M0) for short/medium/long.
4. **Record:** append a dated row per milestone to the `lab_manual.md` results table with
   the before/after ratios and the exit-criteria checklist outcome.

Hold every variable constant except the milestone under test: same host, same model file,
same sequences (`benchmarks/sequences_throughput.fasta`), same warmup/iteration counts.

---

## 8. Risks, dependencies, and rollback

| Risk | Mitigation |
|------|------------|
| Flash attention shifts cosine beyond M8C tolerance | Keep dense path as a runtime-selectable fallback (M2); gate on correctness before merging |
| Graph caching introduces stale-input bugs | Assert on cache key (`n_tokens`); reset all `set_input` tensors every call |
| Thread oversubscription regresses on small machines | Default to physical cores; allow override; benchmark on the target host |
| Quant paths silently upcast weights to F32 | Verify with M0 byte counters that quantized tensors stay quantized on the backend |
| Metal residency change breaks CPU fallback | CI-style run of both backends at every exit |

**Dependency order:** M0 → M1 → (M2, M3 can parallelize) → M4 → M5 → M6.
M1 must land first because graph/weight residency is a prerequisite for clean M2/M3
measurements.

---

## 9. Out-of-scope (explicitly not in this roadmap)

- Algorithmic changes to the ESM-C math (architecture is fixed and validated).
- Multi-GPU / distributed inference.
- Upstreaming to `ggml-org/llama.cpp` (see `ggml/AGENTS.md` AI-contribution policy).
- 600M / 6B model bring-up (tracked separately under M9 in `lab_manual.md`).

---

## 10. Paper update: incorporate performance roadmap results

**Goal.** Revise `paper.tex` so Sections 1, 5, 6, and 7 reflect the completed
M0–M5 + M2.2 milestones, moving flash attention, GPU scheduling, and batching
from "future work" to "implemented" and replacing the M1 throughput/memory
tables with M4 Max data.

**Approach (Option C — Hybrid).** Replace throughput/memory tables with M4 Max
data; keep hardware-independent correctness/downstream as-is; update hardware
description from "16 GB M1" → "M4 Max 36 GB" everywhere.

Dependency order within this section: Phase 0 → Phase 1 → Phase 2.

```
Phase 0 (data)  ─→ Phase 1 (text) ─→ Phase 2 (verify)
   T0              T1–T10              T11–T13
```

Each task has an exit criterion the agent can verify autonomously before
declaring the task complete.

---

### Phase 0 — Gather M4 Max data

#### T0 — Regenerate throughput/memory tables and figures from M4 Max CSV

**What.** Run `benchmarks/throughput.py` and `benchmarks/paper_artifacts.py` on
M4 Max to produce updated throughput/memory CSVs and SVG/PDF figures.

**Exit criterion.**
```bash
rg 'M4 Max' results/throughput_*.csv && ls -t results/paper_artifacts_300m/*.svg | head -5
```

---

### Phase 1 — Text changes (one task per paper section)

#### T1 — Update hardware description in abstract (lines 47–49)

**What.** Change `"a benchmark study on a 16\,GB Apple M1"` → `"a benchmark study
on an Apple M4 Max with 36\,GB unified memory"`. Add brief mention of flash
attention, GPU-resident graphs, and batching as completed contributions.

**Exit criterion.**
```bash
grep -q 'M4 Max' paper.tex && grep -q 'flash attention' paper.tex
```
There must be no remaining mention of "16 GB" or "M1" in the abstract.

---

#### T2 — Update §1 Introduction (lines 58–67)

**What.**
- Line 66: `"16\,GB M1 host"` → `"M4 Max (36\,GB) host"`
- Contribution bullet 4: expand to mention the performance roadmap (flash,
  batching, GPU scheduling) as part of the empirical study.
- Remove or rephrase any "begun to address" language — the roadmap is done.

**Exit criterion.**
```bash
grep -q 'performance roadmap\|flash\|batching\|GPU.*schedul' paper.tex
test $(grep -c '16.*GB.*M1\|M1.*16.*GB' paper.tex) -eq 0
```
Zero instances of "16 GB M1" in the paper.

---

#### T3 — Update §5 hardware description (line 154)

**What.** Change `"Apple M1 (arm64, 16\,GB unified memory, macOS 26.5)"` →
`"Apple M4 Max (arm64, 36\,GB unified memory, macOS 26.5)"`.

**Exit criterion.**
```bash
grep -q 'M4 Max.*arm64.*36.*GB' paper.tex
```

---

#### T4 — Replace §5.2 Throughput table + narrative (lines 178–208)

**What.**
- Replace Table `\label{tab:throughput}` with M4 Max data from EXP-020:
  - F16 Metal: short 10.9ms (91.7 seq/s), medium 27.1ms (36.9 seq/s),
    long 171.9ms (4.9 seq/s)
  - Q4_K_M Metal: short 11.9ms, medium 28.1ms, long 187.8ms
  - Q8_0 Metal: short 12.1ms, medium 30.5ms, long 185.3ms
- Rewrite narrative:
  - F16 is the fastest on M4 Max (native F16 tensor cores)
  - Quantization's value is model-size reduction, not Metal speed
  - CPU path is 10–40× slower than Metal on this hardware
  - Remove "PyTorch MPS remains the fastest" and "0.76× PyTorch CPU long"
    (M1-specific claims)
  - Remove or rewrite "long-sequence collapse...begun to address" — flash
    attention has now landed.

**Exit criterion.**
```bash
# New narrative key indicators:
grep -q 'F16.*fastest\|native.*F16.*tensor' paper.tex && grep -q '10.9\|91.7.*seq/s' paper.tex
# Old M1-specific claims removed:
test $(grep -c 'PyTorch MPS remains the fastest' paper.tex) -eq 0
test $(grep -c 'begun to address' paper.tex) -eq 0
```

---

#### T5 — Add §5.3 Performance Roadmap subsection

**What.** Insert a new subsection after the throughput subsection, describing
the completed performance milestones:
- **M1 (weight/graph residency):** alloc→0, 6× Metal short improvement
  (90ms→15ms)
- **M2 (flash attention):** up to 24% compute reduction at 2002 tokens
- **M3 (GPU scheduler):** 99% alloc reduction, 10–19% compute improvement
- **M2.2 (precision cleanup):** no-op on Metal, forward-compat for CPU

Use numbers from lab_manual EXP-017, EXP-018, EXP-019, EXP-022.

**Exit criterion.**
```bash
grep -q 'weight.*residency\|6.*Metal.*short\|flash.*24.*%\|GPU.*scheduler.*99.*%' paper.tex
grep -q '\\label{sec:perf-roadmap}\|\\label{sec:performance}' paper.tex
```

---

#### T6 — Add §5.4 Batching subsection

**What.** Insert a new subsection after the performance roadmap subsection,
describing the batching results from EXP-021:
- `esmc_embed_batch` API: flat 1D tokens, F16 mask, per-batch graph cache
- Throughput scaling: batch=16 → 3.6× single-seq throughput (345 seq/s)
- Nearly linear to batch=4 (2.8×)
- Key design decisions: post-flash `cont_2d` vs post-dense `permute`-back

**Exit criterion.**
```bash
grep -q '3.6.*batch.*16\|345.*seq/s\|batching' paper.tex && grep -q 'cont_2d\|permute.*back' paper.tex
```

---

#### T7 — Renumber all figures and tables

**What.** After inserting two new subsections (§5.3, §5.4), every
`\label{fig:*}` / `\label{tab:*}` / `Figure~\ref` / `Table~\ref` must be
checked and renumbered to maintain sequence. Update figure captions if the
figure ordering changes.

**Exit criterion.**
```bash
# Verify no duplicate labels:
test $(grep -oP '\\label\{\w+\}' paper.tex | sort | uniq -d | wc -l) -eq 0
```

---

#### T8 — Update §6 Limitations (lines 247–264)

**What.**
- "Scope" bullet: update `"16\,GB M1 host"` → M4 Max
- "Throughput regime" bullet: replace `"Flash attention and persistent
  GPU-resident graphs are the obvious next steps"` with:
  *Flash attention, GPU scheduling, and batching are now implemented and
  measured. The remaining bottleneck is single-hardware scope (36 GB M4 Max
  only) and the 300M-only model restriction.*
- Add a brief note on the batch=32 GPU memory ceiling.

**Exit criterion.**
```bash
# Old text gone:
test $(grep -c 'obvious next steps' paper.tex) -eq 0
# New text present:
grep -q 'implemented.*measured\|batch.*32.*memory.*ceiling\|M4 Max.*36.*GB' paper.tex
```

---

#### T9 — Update §7 Conclusion (lines 265–268)

**What.** Replace `"Extending the runtime to 600M and 6B, adding flash
attention and GPU-resident graphs, and batching for throughput-bound workloads
are the natural next steps"` with:
*Flash attention, GPU scheduling, and batching are now implemented. The
remaining future work is extending the runtime to 600M and 6B, multi-hardware
benchmarking, and optimizing batched throughput beyond batch=16.*

**Exit criterion.**
```bash
# Old text gone:
test $(grep -c 'natural next steps' paper.tex) -eq 0
# New text present:
grep -q 'implemented.*remaining.*future\|600M.*6B.*multi-hardware' paper.tex
```

---

#### T10 — Regenerate figures from M4 Max data

**What.** Replace `figures/throughput_seqps.pdf` and `figures/memory_long_rss.pdf`
with versions generated from M4 Max CSV data. The PDF labels and axis ranges
must be updated to reflect 36 GB hardware.

**Exit criterion.**
```bash
# File is newer than the M4 Max CSV:
test results/paper_artifacts_300m/throughput_seqps.pdf -nt results/throughput_m4_max.csv
test results/paper_artifacts_300m/memory_long_rss.pdf -nt results/memory_m4_max.csv
```

---

### Phase 2 — Verification

#### T11 — Compile paper.tex end-to-end

**What.** Run `pdflatex paper.tex` twice (for cross-references) and confirm
zero errors, zero undefined references.

**Exit criterion.**
```bash
pdflatex -interaction=nonstopmode paper.tex 2>&1 | tail -5 | grep -q 'Output written on paper.pdf'
test $(pdflatex -interaction=nonstopmode paper.tex 2>&1 | grep -c 'Warning.*undefined\|Error\|!') -eq 0
```

---

#### T12 — Verify all cross-references resolve

**What.** Check that every `\ref{...}` / `\cite{...}` points to a defined label.
No `??` in the output PDF.

**Exit criterion.**
```bash
# Check .log for unresolved references:
test $(grep -c 'Rerun to get cross-references right\|undefined.*reference\|Citation.*undefined' paper.log) -eq 0
```

---

#### T13 — Update lab_manual.md with paper change record

**What.** Add a new subsection in the lab_manual's planned-experiments area
(§8) recording which paper sections were changed, which EXP results they
incorporate, and the task completion status.

**Exit criterion.**
```bash
grep -q 'Paper update.*EXP-01[6-9]\|EXP-02[0-2]\|section.*changed\|task.*complete' lab_manual.md
```

---

### Risks

| Risk | Mitigation |
|------|------------|
| M4 Max PyTorch baselines missing from throughput CSV | Re-run `throughput.py` with PyTorch rows enabled before T0 |
| `caisc_2026.sty` not available for compilation | Download from CAISc website before T11 |
| `paper_artifacts.py` CSV column names differ between M1 and M4 Max output | Inspect and align column headers in Phase 0 before figure generation |
| Cross-reference renumbering misses an internal `\ref` | T7 + T12 catch this; T12's exit criterion explicitly checks for `??`
