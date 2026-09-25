Absolutely. I’d make the spec for the **separate ESM-C late-interaction protein retrieval project**, with the paper and software designed together.

# ESMC-Retrieve — Technical Specification

## 1. Project goal

Build an open-source, pip-installable protein retrieval engine that uses **ESM-C residue embeddings + learned late interaction (MaxSim)** to perform fast protein homology / functional similarity search.

The project should target:

* protein homology retrieval
* remote homology detection
* functional similarity search
* motif/domain retrieval
* large local protein databases
* CPU and Apple Silicon inference/search
* millions of protein sequences
* completely local/offline operation

Core idea:

```text
Protein sequence
      │
      ▼
    ESM-C
      │
      ▼
[L × H] residue embeddings
      │
      ▼
projection H → D
      │
      ▼
[L × D] normalized embeddings
      │
      ▼
Late interaction / MaxSim
      │
      ▼
Top-k proteins
```

### Positioning relative to prior work (verified 2026-09)

The model-side idea — residue embeddings + 128D projection + L2 + MaxSim +
InfoNCE on SCOPe/Pfam pairs — was published as **ProtoCol**
(*PROTOCOL: Late Interaction Retrieval for Protein Homolog Search*,
arXiv:2605.29158, May 2026; code available). Cite it, do not claim it.

What is still open, and what this project must deliver instead:

* **Scale/engineering.** ProtoCol explicitly defers efficient late-interaction
  search over large databases. The index (exhaustive → compressed →
  coarse-pruned), throughput/memory numbers, and mmap-backed storage are the
  unclaimed contribution.
* **Stronger backbone.** ProtoCol used ESM-2 35M; ESM-C 300M (H=960) is an
  incremental but reportable upgrade, plus esmc.cpp's Metal/CPU runtime makes
  index builds cheap on Apple Silicon.
* **Benchmark coverage.** ProtoCol evaluates on small group-disjoint test sets
  (SCOPe n=2314, Pfam n=3000) with cRecall@1/10/100 against MinHash, MMseqs2,
  pooled ESM-2 650M, a trained uni-vector ESM-2 35M, and a frozen ProtoCol.
  Missing there: family/superfamily/fold breakdown, identity-controlled splits,
  full-length (non-domain) databases, dense-retrieval SOTA baselines, and any
  scale beyond a few thousand proteins.

ProtoCol facts that shape this plan (verified from the paper, v1 HTML):

* ESM-2 35M (H=480), D=128, fine-tunes the **last 3 blocks** + final LN + W
  (8.4M of 33.6M params trainable); symmetric InfoNCE, τ=1, batch 16,
  3 epochs, lr 2e-5; sequences **truncated to 256**; accidental in-batch
  positives **not filtered**.
* SCOPe cR@10: trained 0.895, **frozen (random-init W, no training) 0.809**,
  pooled ESM-2 650M 0.666, MMseqs2 0.618. Frozen late interaction already
  beats every pooled/alignment baseline — the cheapest experiment in this
  project (frozen ESM-C MaxSim) must therefore run first (M1).
* Code: https://github.com/gabriellecohn/ProtoCol

---

# 2. Relationship with `esmc.cpp`

Keep the projects separate.

### `esmc.cpp`

Responsible for:

* ESM-C model inference
* GGUF model loading
* CPU inference
* Metal inference
* batching
* residue embeddings
* pooling
* low-level model/runtime optimization

### `esmc-retrieval`

Responsible for:

* retrieval model
* projection head
* training
* MaxSim
* indexing
* vector storage
* quantization
* approximate/exhaustive search
* Rust acceleration
* Python API
* benchmarks
* datasets/evaluation
* paper experiments

Dependency:

```text
esmc-retrieval
       │
       └── esmc.cpp
```

Do **not** copy the ESM-C implementation into the new repository.

### Python bridge (must be specified)

`esmc.cpp` exposes only a CLI and a C API (`src/esmc.h`: `esmc_tokenize`,
`esmc_embed`, `esmc_embed_batch`, `esmc_embed_mean`) — **no Python bindings
exist today**, and CMake currently builds `esmc` as a **static** library only
(`CMakeLists.txt:16`).

**Laziest path first (M0–M5): no bridge at all.** The existing CLI already does
what index building needs: `esmc-embed --fasta in.fa --output-dir cache/` loads
the model once, uses the length-aware batch schedule, and writes one
`[n_res, 960]` `.npy` per sequence **with BOS/EOS already stripped**
(`examples/embed/main.cpp:216-226`). Fine for SCOPe/Pfam-sized sets
(10³–10⁵ files); unworkable at 10⁶ files (M6 fixes that).

A real bridge is needed only for interactive `index.search(sequence)` (M8).
Options:

* **cffi over a new shared build** — add a `SHARED` target in esmc.cpp, load
  `libesmc` from Python. FFI surface stays one header.
* **PyO3 extension linking the static lib** — one wheel, no runtime .so to
  ship, but the retrieval repo owns build tooling for a C++ dependency.

API facts any bridge must handle:

* `esmc_embed` returns **`[n_tokens, 960]` including BOS and EOS rows**;
  residues are rows `1 .. n_tokens-2` (the CLI strips them the same way,
  `examples/embed/main.cpp:95`). A one-row off-by-one silently embeds
  `<cls>`/`<eos>` as if they were residues.
* `esmc_embed_batch` writes a **packed** buffer `[Σ lengths_i, 960]` —
  sequence *s* starts at row `Σ_{t<s} lengths_t`, and each sequence still
  carries its own BOS/EOS rows. There are no pad rows in the output
  (`examples/embed/main.cpp:305-310`).
* Context length is 2048 tokens (`esmc_n_ctx`, GGUF `esmc.context_length`),
  i.e. 2046 residues after BOS/EOS; see §5 for long proteins.
* Use **f16 or q8_0** GGUFs only. q4_k_m / q4_k_s fail esmc.cpp's own
  correctness gate on some sequences (2 / 13 of 100); an index built on them
  would silently degrade.

Dependency direction: the **Rust/numpy search core** never touches esmc.cpp.
The **Python layer** needs esmc.cpp at build time (DB embeddings) *and* at
query time (every query sequence must be embedded).

---

# 3. Proposed repository

```text
esmc-retrieval/
│
├── python/
│   └── esmc_retrieval/
│
├── rust/
│   └── retrieval/
│
├── training/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── train.py
│   └── configs/
│
├── benchmarks/
│   ├── retrieval/
│   ├── latency/
│   ├── memory/
│   └── scripts/
│
├── examples/
│   ├── basic_search.py
│   ├── fasta_search.py
│   └── build_index.py
│
├── tests/
│
├── docs/
│
├── pyproject.toml
├── Cargo.toml
├── README.md
└── LICENSE
```

This is the *eventual* layout. Start with only `pyproject.toml`,
`esmc_retrieval/`, `scripts/`, `tests/`, `results/`. Add `rust/` + `Cargo.toml`
only if the M5 gate (T5.3) proves a fused kernel is worth it; add
`training/configs/` only when there is more than one config to keep.

---

# 4. Python API

The API should be extremely simple.

### Basic search

```python
from esmc_retrieval import ProteinIndex

index = ProteinIndex.from_fasta(
    "proteins.fasta",
    model="esmc-300m"
)

results = index.search(
    "MKTIIALSYIFCLVFADYKDDDDK",
    k=20
)
```

Return:

```python
[
    {
        "id": "protein_123",
        "score": 0.87,
        "metadata": {...}
    },
    ...
]
```

### Persistent index

```python
index = ProteinIndex.build(
    "proteins.fasta",
    output="proteins.esmci"
)

index = ProteinIndex.load("proteins.esmci")

results = index.search(sequence, k=20)
```

### Batch search

```python
results = index.search_batch(
    sequences,
    k=20
)
```

---

# 5. Retrieval model

Initial architecture:

```text
ESM-C
  │
  │ residue embeddings
  ▼
Linear(H → 128)
  │
  ▼
L2 normalization
  │
  ▼
residue vectors
```

For proteins A and B:

```text
A = [a1, a2, ..., am]
B = [b1, b2, ..., bn]
```

Compute:

$$
S(A,B)=\sum_i \max_j(a_i \cdot b_j)
$$

This is the central retrieval mechanism.

### Optional symmetric score

Evaluate:

$$
S_{sym}(A,B)
=
\frac{1}{2}
[
S(A,B)+S(B,A)
]
$$

This should be an experiment rather than hard-coded initially.

### Suspected confound: MaxSim length bias (measure it, M1/M4)

`max_j` over more candidate residues is stochastically larger, so
`S(A,B)` should grow with `len(B)` regardless of homology: a 2000-residue
multi-domain protein could accumulate more MaxSim mass than a 300-residue
true homolog. Treat this as a **hypothesis to measure** (Spearman ρ between
candidate length and score over non-homolog pairs), not an established fact.
`S_sym` does not fix it — both terms are still unnormalized.

It is likely to be *invisible on SCOPe*: domains are short and fairly uniform,
and ProtoCol truncates to 256 residues. It will matter on full-length
UniProt chains, which is what users will actually index — hence M4.

Normalizations to compare (ablate in §16, do not hard-code):

```text
raw sum          Σ_i max_j (a_i · b_j)          — baseline (ProtoCol)
÷ query len      S(A,B) / len(A)                — rank-invariant per query;
                                                  only matters for calibration
÷ both lens      S_sym / (len(A) + len(B))      — symmetric normalized
÷ √(len_A·len_B) geometric length penalty       — our variant, not from ColBERT
```

Note `÷ len(A)` cannot change a single query's ranking (constant per query);
it only makes scores comparable across queries. The two variants that touch
`len(B)` are the ones that can actually fix candidate-length bias.

If raw sum wins on recall but loses calibration, report both: recall and
score-vs-length correlation.

### Long proteins (>2046 residues)

ESM-C context is 2048 tokens = 2046 residues + BOS/EOS. DB proteins can exceed
it (human titin, Q8WZ42, is 34,350 aa).
Plan: chunk to overlapping windows, embed each chunk as an independent
residue set, concatenate residue rows (chunk identity must be dropped from
MaxSim — a residue matches the best residue anywhere in the candidate).
Truncating instead would silently lose C-terminal domains. Record how many
DB proteins are affected; it is a small fraction but they are precisely the
interesting multi-domain cases for the length-bias analysis.

---

# 6. Training

## Phase 1 — baseline

Freeze ESM-C.

Train only:

```text
ESM-C → Linear(H,128)
```

Advantages:

* very cheap
* establishes whether the representation works
* easy to reproduce
* avoids expensive 300M-parameter training

---

## Phase 2 — partial fine-tuning

Fine-tune:

```text
last 2–3 ESM-C blocks
+
projection head
```

Compare against frozen baseline.

---

## Phase 3 — parameter-efficient training

Investigate:

```text
ESM-C
 +
LoRA
 +
projection head
```

This is particularly attractive for researchers with limited GPU resources.

**Gated, not planned.** LoRA only runs if Phase 2 beats Phase 1 *and* memory
is the limiter (M3). ProtoCol's recipe is already Phase 2.

### Train/serve skew (the non-obvious constraint)

esmc.cpp is inference-only; gradients need PyTorch ESM-C (`esm` package).

* **Phase 1 (frozen):** embed once *with esmc.cpp* and train the head on that
  cache. Training sees byte-for-byte the embeddings the index will store —
  zero skew, and each epoch is seconds–minutes on a laptop.
* **Phase 2/3 (fine-tuned):** the updated weights must be re-exported to GGUF
  (`tools/convert_esmc_to_gguf.py`) and pass esmc.cpp's correctness gate
  against the fine-tuned PyTorch model before any index is built. Skipping
  this means training one model and serving another.
* Trick for Phase 2: cache the *input to block 30−k* once; fine-tuning the
  last k blocks then only runs k blocks per step.
* Layer-choice ablations (e.g. mean of top layers) are **PyTorch-only**:
  esmc.cpp exposes final-layer outputs only.

---

# 7. Training objective

Use contrastive learning.

For a query protein `q`:

```text
q
├── positive homolog
└── negatives
```

Score:

```text
s(q,p)
```

Then use InfoNCE:

$$
L =
-\log
\frac{\exp(s(q,p^+)/\tau)}
{\sum_j\exp(s(q,p_j)/\tau)}
$$

Start with:

```text
temperature = 1.0
in-batch negatives = yes
```

Later test hard negatives.

### Two required details

**False-negative masking.** With SCOPe/Pfam *homology* positives, an
"in-batch negative" can be an unlabelled homolog of the query — in-batch
negatives are exactly where this happens. Mask any in-batch pair sharing a
superfamily/clan label with the positive before the denominator. (ProtoCol did
not filter these; it is a cheap correctness win and a reportable difference.)

**Temperature.** τ=1.0 is defensible for *summed* MaxSim — logits are sums
over query residues, so they are O(len) not O(1), and ProtoCol used τ=1 with
the same score. For a single-vector baseline the literature default is
0.05–0.1 (CLIP starts at 0.07). Sweep τ ∈ {0.03, 0.1, 0.3, 1.0} for both
score types (§16) rather than assuming one value carries over.

---

# 8. Positive pairs

Use biologically meaningful relationships rather than random sequence similarity.

Potential hierarchy:

### Primary

SCOPe superfamily relationships.

### Secondary

Pfam family/clan relationships.

### Additional evaluation

Hold out sequence/family relationships to test remote homology rather than memorization.

Important:

**Train/test splits must prevent homolog leakage.**

This becomes an important methodological contribution of the paper.

---

# 9. Retrieval backends

Build the system in layers.

### V0 — brute force

```text
query
 ↓
every protein
 ↓
MaxSim
 ↓
top-k
```

This establishes the scientific baseline.

### V1 — optimized exhaustive search

Rust implementation with:

* SIMD
* multithreading
* batching
* cache-friendly memory layout
* tiled matrix operations

### V2 — compressed representation

Experiment with:

* FP16
* INT8
* product quantization

### V3 — approximate retrieval

Potentially investigate:

```text
coarse retrieval
      ↓
candidate proteins
      ↓
exact MaxSim reranking
```

Don't implement ANN prematurely. First demonstrate that optimized exhaustive search is already useful — that means V0/V1 delivering honest numbers at 10K–1M (see the scaling table in §15).

**V3 is not optional past ~1M.** The §15 cost table shows exhaustive search
at 10M+ costs minutes per query, so the 10M/100M scaling rows and the
"millions of sequences" goal in §1 both depend on this stage. Design V1's
memory layout with the V3 coarse stage in mind (one extra fixed-size vector
per protein — pooled D-dim or a Muvera FDE — stored alongside the residue
vectors), so adding it later is a new scorer, not a new format.

Use existing coarse-stage algorithms rather than inventing one: **PLAID**
(ColBERTv2 engine), **Muvera** fixed-dimensional encodings, and **DESSERT**
are all published multi-vector retrieval methods (all cited by ProtoCol).

Also worth pricing before building: **FAISS flat** on pooled vectors (latency
floor) and existing MaxSim implementations (**PyLate**; `maxsim-rs` pending
verification) are the engineering baselines (§15). If a fused Rust kernel
cannot beat them on proteins/sec, the systems contribution needs a different
angle (index format + mmap + esmc.cpp-integrated build pipeline), not a
faster matmul.

Why a fused kernel *might* matter (to be measured in M5, not assumed): MaxSim
is a GEMM followed by a max. Unfused, the `[L_q × tile]` score block is
written to memory before the max; at L_q=300 with FP32 scores that is ~4.7×
more bytes than the FP16 DB vectors being read. Fusing the max into the GEMM
tile removes that traffic. If BLAS/MPS tiled search already hits >50% of GEMM
peak, the kernel is not worth writing.

---

# 10. Rust architecture

Python:

```text
ProteinIndex
     │
     ▼
PyO3
     │
     ▼
Rust retrieval engine
```

Rust responsibilities:

```text
Index
 ├── sequence metadata
 ├── residue embeddings
 ├── memory mapping
 ├── quantization
 ├── SIMD MaxSim
 ├── multithreading
 └── top-k
```

Potential internal API:

```rust
search(
    query_embeddings,
    k,
    options
) -> Vec<SearchResult>
```

The Rust layer should never know anything about Python objects.

---

# 11. Index format

Create a custom simple binary format.

Example:

```text
proteins.esmci
│
├── header
├── metadata
├── sequence offsets
├── embedding offsets
├── embeddings
└── optional quantization data
```

Use memory mapping:

```text
mmap
  ↓
embedding storage
  ↓
Rust SIMD search
```

This allows databases larger than RAM to potentially be handled efficiently.

---

# 12. Database metadata

Each entry should support:

```json
{
    "id": "P12345",
    "description": "...",
    "length": 431,
    "source": "uniprot",
    "taxonomy": "Homo sapiens"
}
```

But retrieval itself should remain independent of metadata.

---

# 13. CLI

Provide a dead-simple CLI.

### Build

```bash
esmc-retrieval build proteins.fasta \
    --output proteins.esmci
```

### Search

```bash
esmc-retrieval search proteins.esmci \
    --sequence "MKT..." \
    --top-k 20
```

### FASTA query

```bash
esmc-retrieval search proteins.esmci \
    --query queries.fasta \
    --output results.tsv
```

### Benchmark

```bash
esmc-retrieval benchmark proteins.esmci
```

---

# 14. Hardware targets

First-class targets:

### Apple Silicon

```text
M1
M2
M3
M4
```

CPU + Metal ESM-C inference through `esmc.cpp`.

### Linux CPU

AVX2 initially.

Later:

```text
AVX-512
ARM NEON
```

GPU retrieval can come later.

---

# 15. Benchmarks

The paper needs three classes of benchmarks.

## Scientific

Compare:

```text
ESM-C pooled embedding
vs
ESM-C + MaxSim
vs
ESM-C + trained projection + MaxSim
vs
alignment-based methods
```

Metrics:

* Recall@1
* Recall@5
* Recall@10
* MRR
* ROC-AUC / PR-AUC where appropriate
* remote-homology retrieval performance

### Required baselines (the plan's original list is not reviewable)

The four above only compare *ourselves against ourselves* plus a generic
"alignment method". A paper needs:

```text
Direct prior work (verified)
  ProtoCol (arXiv:2605.29158) — same idea, ESM-2 35M. Run their code (M0/T0.7).

Retrieval SOTA (dense)
  DHR (Nat Biotech 2024)      — contrastive dual-encoder PLM retriever
  TM-Vec + DeepBLAST (2024)   — twin net on residue embeddings
  PLMSearch / PLMAlign (Nat Commun 2024) — similarity head + residue alignment
  ERAST (reported Nat Biotech 2026, billion-scale) — UNVERIFIED, check in T0.2

Residue-level but alignment-aggregated (isolates MaxSim vs alignment)
  pLM-BLAST (Bioinformatics 2023) / PEbA (2024)

Baselines we already have easy access to
  knnProtT5 (Schütze 2022) / mean-pooled ESM-C (cosine)
  MMseqs2 -s 7.5, MinHash 5-mer (ProtoCol's exact settings), BLAST
  Foldseek (where SCOPe has structures) — cite published SCOPe40 numbers

Index/engineering baseline
  FAISS flat on pooled vectors; PyLate (CIKM 2025) retrieval;
  maxsim-rs Rust crate — UNVERIFIED, check in T0.2
```

Citations marked UNVERIFIED came from a literature sweep and were not
confirmed against a primary source; they do not go in the paper until T0.2
resolves them.

Do not claim the idea against ProtoCol; claim the *index* against FAISS and
the *coverage* against everyone else.

### Evaluation protocol additions

* Report ProtoCol's **capped recall** cR@{1,10,100} (their eq. 4) so numbers
  are directly comparable, plus **sensitivity up to the first false positive**
  at family/superfamily/fold — the convention used by MMseqs2/Foldseek-style
  SCOPe benchmarks, so published numbers can be cited instead of re-run.
* Report recall separately at **family / superfamily / fold** level (the
  standard SCOPe breakdown), not one aggregate number.
* Add **sequence-identity-controlled** results on ASTRAL subsets
  (10/20/30/40% identity) — this is what makes a "remote homology" claim
  credible; superfamily-aggregate recall alone does not.
* Leakage-safe splits remain mandatory (§8); state the identity threshold
  used between train and test, not just "no homolog leakage".
* Report score-vs-length correlation alongside recall, so the MaxSim length
  bias (§5) is visible rather than baked into the headline number.

---

## Systems

Measure:

```text
proteins/sec
queries/sec
latency/query
index build time
memory usage
database size
```

Compare:

```text
Python
vs
Rust
vs
Rust + SIMD
vs
Rust + quantization
```

---

## Scaling

Test approximately:

```text
10K → 100K → 1M      (exhaustive, honest)
10M → 100M           (only with the V3 coarse stage; see below)
```

**Exhaustive MaxSim cost.** Per query: `L_q × D × N_db_residues` MACs
(one big matmul + max-reduce), 2 FLOPs per MAC. With D=128, L_q=300,
~400 residues/protein:

| DB size | DB residues | MACs/query | FP16 index (D=128) | Search/query @ 1 TFLOP/s* | DB build @ 7.2k res/s† |
|--------:|------------:|-----------:|-------------------:|--------------------------:|-----------------------:|
| 10K  | 4×10⁶  | 1.5×10¹¹ | 1.0 GB | 0.3 s  | 9 min |
| 100K | 4×10⁷  | 1.5×10¹² | 10 GB  | 3 s    | 1.5 h |
| 1M   | 4×10⁸  | 1.5×10¹³ | 102 GB | 30 s   | 15.5 h |
| 10M  | 4×10⁹  | 1.5×10¹⁴ | 1 TB   | 5 min  | 6.5 days |
| 100M | 4×10¹⁰ | 1.5×10¹⁵ | 10 TB  | 50 min | 65 days |

\* Placeholder sustained throughput; M5/T5.2 measures the real CPU and GPU
peak on the target machine and replaces this column.
† Measured: esmc.cpp `--fasta`, mixed 1000-seq corpus, M4 Max Metal F16,
length-aware batch schedule = 7,161 residues/s end-to-end.

So:

* **Search:** exhaustive is interactive to ~100K; 1M is batch-mode on CPU
  (possibly interactive on the GPU — measure, T5.2).
* **Build dominates at scale.** Embedding the DB, not searching it, is the
  wall: ~16 h per 1M proteins on one M4 Max, ~1 week for 10M, ~2 months for
  100M. 100M is out of scope on a laptop regardless of search algorithm;
  10M is a "one machine-week" experiment that must be budgeted explicitly.
* **10M/100M rows motivate V2/V3**, they are not promised results. Do not ship
  an unqualified "we searched 100M proteins" claim.

Order: V0/V1 measurements (10K–1M) first, V3 second, 10M only if the
build budget is accepted at the M7 gate.

---

# 16. Critical ablations

At minimum:

### Representation

```text
pooled ESM-C
vs
residue ESM-C
```

### Interaction

```text
mean cosine
vs
global cosine
vs
MaxSim
```

### Projection

```text
none
256D
128D
64D
```

### Fine-tuning

```text
frozen
projection only
last 2 layers
last 3 layers
LoRA
```

### Precision

```text
FP32
FP16
INT8
```

### Search

```text
Python
Rust
Rust + SIMD
```

### Scoring / training (added after plan review)

```text
Length normalization   raw sum vs ÷(len_A+len_B) vs ÷√(len_A·len_B)   (M1, M4)
Temperature            τ ∈ {0.03, 0.1, 0.3, 1.0}                     (M2)
Negatives              in-batch raw vs false-negative-masked          (M2)
                       hard negatives only if masked wins
Long proteins          truncate-256 (ProtoCol) vs chunk-and-concat    (M4)
Backbone               ESM-C 300M vs ESM-2 35M (ProtoCol's setting)   (M0/M1)
Layer choice           final vs mean of top layers — PyTorch-only,
                       esmc.cpp exposes final layer only; drop unless cheap
```

---

# 17. Paper hypothesis

The central hypothesis should be:

> **Residue-level late interaction over ESM-C representations can provide more sensitive protein retrieval than pooled embeddings while remaining computationally practical through low-dimensional projection and systems-level optimization.**

This gives you both a **scientific contribution** and a **systems contribution**.

Reframed after the prior-art check (§1): the *sensitivity* half of this is
ProtoCol's result already. The testable, still-open version is:

> **Late-interaction protein retrieval stays practical at database sizes where
> pooled-vector and alignment baselines stop being practical — and the
> sensitivity advantage holds for a stronger backbone (ESM-C) with properly
> controlled benchmarks (identity splits, length-normalized scores).**

---

# 18. Paper contributions

Target 4 contributions:

### 1. Representation

~~Adapt ESM-C for residue-level late-interaction retrieval.~~
**Rework:** ProtoCol already adapted residue-level late interaction (ESM-2 35M);
the contribution here is (a) ESM-C 300M as backbone with a controlled
backbone ablation — including the possibility (tested first, M1) that a
stronger frozen backbone makes contrastive training unnecessary — and
(b) measuring MaxSim's suspected length bias on full-length databases and
correcting it if it exists (M4). ProtoCol truncates to 256 residues and
evaluates on domains, so it could not have observed it.

### 2. Retrieval

Demonstrate MaxSim-based protein retrieval for remote homology/function-related
search — **against ProtoCol, DHR, TM-Vec, PLMSearch, pLM-BLAST and Foldseek**,
with family/superfamily/fold breakdown and identity-controlled splits, rather
than against pooled-only baselines.

### 3. Systems

Develop a Rust SIMD implementation enabling scalable local retrieval —
the genuinely open half: **ProtoCol explicitly defers efficient large-scale
search**. Deliver exhaustive search to ~10⁶ proteins with measured
proteins/sec, plus the coarse→rerank stage beyond that.

### 4. Software

Release an open-source pip-installable, offline protein retrieval system.

The combination is still stronger than a bare embedding release — but only if
the paper is written **relative to ProtoCol**, not relative to an imaginary
baseline of "just another protein embedding model".

---

# 19. Milestones (agent-loop ready)

Order: **M0 → M1 → M2 → (M3, gated) → M4 → M5 → M6 → M7 → M8 → M9.**
M5 may start in parallel after M1 using synthetic vectors (systems numbers do
not depend on the trained model).

## 19.0 Loop rules (apply to every task)

1. **One eval harness, frozen after M0.** Every number comes from
   `esmc_retrieval/eval.py` at a recorded split hash. Never select on test;
   val = superfamilies/clans carved out of train.
2. **Every run appends one line to `results/runs.jsonl`:** git sha, task ID,
   config, seed, metrics, wall time, hardware, model file hash.
3. **Exit criteria are commands or files, not opinions.** A task is done when
   its check passes. "Looks good" is not an exit.
4. **Budget caps.** Hitting a task's budget without meeting its exit = stop and
   report, do not keep iterating.
5. **Gates are decisions.** At each gate write `docs/decisions/Mx.md`
   (≤10 lines: result, decision, why). A negative result is recorded and the
   plan branches; it is not retried quietly.
6. **Baselines before models, dumb before smart, frozen before trained.** One
   change at a time; every new method is compared to the previous best under
   the same harness.
7. **Look at the data** every milestone (failure cases, similarity maps,
   length histograms), not only aggregate metrics.

---

## M0 — Ground truth: data, splits, metric, dumb baselines (no model code)

Goal: an eval harness whose numbers you would bet on, with ProtoCol reproduced
on it. Budget: ~2 days; CPU, plus one GPU/MPS run for T0.7.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T0.1 | Minimal repo: `pyproject.toml`, `esmc_retrieval/`, `scripts/`, `tests/`, `results/`. No Rust, no configs. | repo | `pip install -e . && pytest` passes (one smoke test) |
| T0.2 | Verify every citation in §15: DOI/arXiv resolves and the claimed fact matches the source. Resolve ERAST and `maxsim-rs`. | `docs/related_work.md` | each row has URL + one verified sentence; no UNVERIFIED rows remain (resolve or delete) |
| T0.3 | Fetch SCOPe 2.08 ASTRAL 40% + Pfam clan mapping; pin versions and sha256. | `scripts/fetch_data.py`, `data/manifest.json` | second run is a no-op; checksums match; domain/superfamily/fold counts recorded |
| T0.4 | Group-disjoint splits (train/val/test by superfamily; by clan for Pfam). If ProtoCol's repo ships its split, keep it as `test_protocol`. | `data/splits/*.tsv` + hash | test asserts zero group overlap between splits; MMseqs2 max train↔test identity reported |
| T0.5 | Metrics: cR@{1,10,100} (ProtoCol eq. 4), MRR, sensitivity-to-1st-FP at family/superfamily/fold. | `esmc_retrieval/eval.py` | unit tests on hand-computed toy rankings pass; random ranking lands within 3σ of its analytic expectation |
| T0.6 | Dumb baselines: random, MinHash 5-mer (256 perms, datasketch), MMseqs2 `easy-search -s 7.5`, ranked by e-value. | `results/m0_baselines.csv` | all three on the full test split, logged |
| T0.7 | Reproduce ProtoCol with their code. | `results/m0_protocol_repro.csv` | trained SCOPe cR@10 within ±0.02 of 0.8947, **or** a written gap analysis plus a rerun of their code on our split, which becomes the baseline |

**Gate M0.** Harness frozen (`eval.py` + split hash committed). If MinHash or
MMseqs2 are >0.05 off ProtoCol's Table 1 on the same split, stop: the harness
is wrong, not the baselines.

---

## M1 — Frozen ESM-C, zero training (the cheapest scientific result)

Why first: ProtoCol's frozen variant (random 128D projection, no training)
already reached SCOPe cR@10 0.809 vs 0.666 for pooled ESM-2 650M. Frozen
ESM-C 300M residue embeddings may land near *trained* ProtoCol. If so, that
is a headline result that costs no training. Budget: ~1 day.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T1.1 | Embed all SCOPe domains with the existing CLI: `esmc-embed -m esmc-300m-f16.gguf --fasta scope.fa --output-dir cache/` (strips BOS/EOS, loads once, length-aware batching). f16 or q8_0 only. | `cache/*.npy`, wall time | file count = domain count; every shape `[len, 960]`; 10 random domains vs PyTorch `esm` ESMC: min cosine ≥ 0.999 |
| T1.2 | Pooled ESM-C (mean + cosine). | results row | logged |
| T1.3 | Raw MaxSim at 960D (no projection) and random-projection 128D (ProtoCol-F analogue). Exact, tiled torch/numpy. | results rows | 50 pair scores match a naive double loop to 1e-5 |
| T1.4 | Length bias: Spearman ρ(candidate length, score) over non-homolog pairs for each scorer; cR@k for each normalization in §5. | `results/m1_length_bias.csv` | ρ and cR@k per scorer logged |
| T1.5 | Look at the data: 20 rank-1 false positives (lengths, class, fold) and 3 residue similarity maps. | `docs/m1_failures.md` + plots | every failure has a one-line hypothesis |

**Gate M1.**
* Frozen raw MaxSim (ESM-C) ≥ trained ProtoCol on cR@10 → training becomes an
  ablation, not the method. M2 shrinks to T2.1–T2.4; skip M3.
* Frozen MaxSim < pooled ESM-C → something is broken (BOS/EOS, normalization,
  eval). Do not start M2 until explained.

---

## M2 — Train the projection head on the cached embeddings

Backbone frozen, so train on the M1 cache: no train/serve skew (§6), and an
epoch takes seconds–minutes on a laptop. Budget: ~3 days.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T2.1 | Symmetric InfoNCE over MaxSim (ProtoCol eq. 3). Check PyLate first and reuse it if it accepts precomputed embeddings; otherwise ~100 lines of PyTorch. | `esmc_retrieval/train.py` | one step runs on CPU and MPS |
| T2.2 | Sanity check: overfit one batch of 16 pairs. | log | loss < 0.05 within 500 steps |
| T2.3 | False-negative masking: same superfamily/clan inside a batch is excluded from the denominator. | unit test | masked logits are −inf; with no collisions, loss equals the unmasked reference |
| T2.4 | Train D=128: τ ∈ {0.03, 0.1, 0.3, 1.0} × {masked, unmasked}, best M1 normalization, 3 seeds. Select on val only. | `results/m2_sweep.csv` | best config as mean ± std over 3 seeds; test evaluated **once**, after selection |
| T2.5 | D ∈ {64, 128, 256}. | results rows | table logged, with index bytes per residue |

**Gate M2.** Trained head beats frozen raw MaxSim on val by >2σ across seeds →
keep it. Otherwise record the negative result and ship frozen (reduce to 128D
by PCA only for index size).

---

## M3 — Partial fine-tuning (gated: skip unless M2 leaves headroom)

Budget: ~1 week of one GPU (or ~2–3× that on MPS).

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T3.1 | PyTorch ESM-C: fine-tune the last k ∈ {2, 3} blocks + head (ProtoCol's recipe). Cache inputs to block 30−k once. | `train.py --finetune k` | with k blocks unmodified, outputs match esmc.cpp embeddings: min cosine ≥ 0.999 |
| T3.2 | Train with 3 seeds, same selection protocol as T2.4. | results | gain over best M2 >2σ, else stop here (LoRA dropped) |
| T3.3 | Export to GGUF (`tools/convert_esmc_to_gguf.py`); validate with esmc.cpp `benchmarks/correctness.py` against the fine-tuned PyTorch model. | `esmc-300m-ft-f16.gguf` | 100/100 pass, min cosine ≥ 0.999 |

---

## M4 — Deployment-shaped benchmark (full-length chains)

SCOPe domains are short and single-domain, and ProtoCol truncates to 256.
A UniProt database is the opposite. Length bias and long-protein handling only
show up here. Budget: ~3 days.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T4.1 | Full-length DB: ~50–100K Swiss-Prot chains with Pfam clan labels; test clans disjoint from training clans. | `data/fulllength/` + manifest | length histogram; count of chains >2046 aa recorded |
| T4.2 | Chunk sequences >2046 aa: overlapping windows, concatenate residue rows, drop duplicated overlap rows. | `embed.py` + test | output row count == sequence length exactly; no special-token rows |
| T4.3 | Eval with query = domain or chain, DB = full chains; every scorer/normalization; ρ(length, score). | `results/m4_fulllength.csv` | table + ρ per scorer |

**Gate M4.** Freeze the default scorer (model + D + normalization + chunking).
Everything after this point uses it unchanged.

---

## M5 — Exhaustive search engine (measured, not assumed)

Budget: ~1 week.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T5.1 | Reference exact search: tiled GEMM + segment-max + top-k in torch/numpy, CPU and MPS. | `esmc_retrieval/search.py` | top-k IDs identical to a naive loop on a 1K-protein DB |
| T5.2 | Measure q/s at 10K and 100K, plus raw GEMM peak on the same device. Efficiency = achieved / peak FLOP/s. Replace the placeholder column in §15. | `results/m5_search.csv` | table includes efficiency %; §15 updated with measured values |
| T5.3 | **Gate before Rust:** only if T5.2 efficiency <50%, write a fused kernel (GEMM tile + running max; score block never materialized) in Rust/PyO3 or Metal. | `rust/` | same top-k IDs as T5.1 and ≥2× T5.1 throughput at 100K; otherwise delete `rust/` and ship T5.1 |
| T5.4 | Engineering baselines: FAISS flat on pooled vectors (latency floor); PyLate retrieval; `maxsim-rs` if T0.2 verified it. | results | comparison table logged |

---

## M6 — Index format and build pipeline

Budget: ~1 week.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T6.1 | Single-file embedding output (10⁶ `.npy` files is unworkable): add a packed-output mode to `esmc-embed` (small esmc.cpp PR), **or** a cffi bridge over a shared `libesmc`. Prefer the CLI flag unless M8 needs the bridge anyway. | esmc.cpp PR or `bridge.py` | 10K proteins → one file; values equal the per-file `.npy` path after f16 cast |
| T6.2 | `.esmci`: header (magic, version, D, dtype, n_proteins, n_residues) + u64 offsets + projected embeddings (+ optional coarse vectors). Metadata as a Parquet/JSONL sidecar, not a custom binary format. mmap load. | `esmc_retrieval/index.py` + `docs/esmci.md` | round-trip test; a 100K index opens in <1 s; results identical to in-memory search |
| T6.3 | Project at build time (store D, not 960: 7.5× smaller at D=128). | — | file size = n_res × D × 2 B + header + offsets (±1%) |
| T6.4 | Measure end-to-end build rate on M4 Max. | results | proteins/s logged; §15 build column replaced with measured values |

---

## M7 — Compression and coarse stage (past ~1M)

Budget: ~2 weeks, plus the explicit build budget for any 10M run.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T7.1 | INT8 and ColBERTv2-style residual compression. | results | cR@10 drop ≤0.01 at ≥4× size reduction, else documented as a negative result |
| T7.2 | Coarse candidates (Muvera FDE or pooled vector + FAISS) → exact MaxSim rerank. Use published algorithms; do not invent one. | `esmc_retrieval/coarse.py` | candidate-set recall vs exhaustive ≥0.99 at the chosen C; latency at 1M logged |
| T7.3 | Scaling: 10K / 100K / 1M (UniRef50 subsets). 10M only if the ~1 machine-week build is approved at this gate. | `results/m7_scaling.csv` | build time, index size, latency, and recall-vs-exhaustive for each size |

---

## M8 — Package and CLI

Budget: ~1 week.

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T8.1 | `ProteinIndex.from_fasta / build / load / search / search_batch`; CLI `build / search / benchmark` (§4, §13). Query embedding through esmc.cpp (bridge or CLI). | package | fresh venv, network off: `pip install .` → build a 1K index → search → correct top-1 on 10 known homolog pairs |
| T8.2 | Tests: BOS/EOS strip, >2046 aa chunking, index round-trip, deterministic top-k, q4 GGUF rejected or warned. | `tests/` | `pytest` green on macOS arm64 and Linux x86 CI |

---

## M9 — Paper

| ID | Task | Artifact | Exit criterion |
|---|---|---|---|
| T9.1 | One command regenerates every table and figure from `results/runs.jsonl` at a pinned commit. | `make tables` | regenerated tables diff to zero against the paper |
| T9.2 | Claims audit: each numeric sentence maps to a table cell; ProtoCol framed as prior work for the model idea (§18). | `docs/claims.md` | 100% of claims mapped; no UNVERIFIED citation in the paper |

Then freeze the API and publish the package.

---

# 20. What I would explicitly NOT build initially

Avoid scope creep.

Don't initially build:

* web UI
* hosted API
* database server
* GPU retrieval
* distributed indexing
* complicated ANN infrastructure
* protein generation
* structure prediction
* PPI prediction
* functional annotation model
* giant training pipeline

The core product should remain:

> **`pip install` → give it a protein database → search proteins locally.**

---

# 21. Suggested project name

For now:

**`esmc-retrieval`**

Possible paper title (must not read as a re-title of ProtoCol's *"Late
Interaction Retrieval for Protein Homolog Search"*, arXiv:2605.29158 — lead
with the differentiator, which is scale/systems):

> **Scaling Late-Interaction Protein Retrieval: a Practical Index for ESM-C Residue Embeddings**

Possible library description:

> **Fast local protein retrieval using ESM-C residue embeddings and late interaction.**

---

## 22. Definition of done

I'd consider v1 successful when this works:

```python
from esmc_retrieval import ProteinIndex

index = ProteinIndex.from_fasta(
    "uniprot_subset.fasta"
)

hits = index.search(
    query_sequence,
    k=20
)
```

and internally:

```text
FASTA
 ↓
ESM-C
 ↓
residue embeddings
 ↓
trained 128D projection
 ↓
L2 normalization
 ↓
Rust MaxSim
 ↓
top-k
```

with reproducible evidence that **the learned residue-level retrieval model provides useful biological retrieval performance and the Rust implementation makes search practical at substantially larger database sizes.**

Concretely, "done" means all of:

1. `search()` works and beats **mean-pooled ESM-C cosine** and **MMseqs2** at
   superfamily-level recall — and is reported **next to ProtoCol**, citing it
   for the model idea (§1, §18).
2. Recall broken out at family/superfamily/fold, plus identity-controlled
   (ASTRAL) numbers — not one aggregate (§15).
3. Length bias either fixed or quantified: recall reported with and without
   score normalization (§5, §16).
4. A measured systems claim at a stated DB size with stated latency — e.g.
   "N queries/sec at 1M proteins on M4 Max CPU", with the exhaustive/exact
   qualifier where it applies (§15 scaling table).
5. `pip install` → build index → search, offline, with the BOS/EOS and
   >2046-aa cases covered by a test (§2, §5; T8.2).
6. Every gate M0–M7 has a `docs/decisions/Mx.md`, and every paper number
   regenerates from `results/runs.jsonl` (T9.1).

This is the spec I'd use as the **master technical spec** before writing code.
