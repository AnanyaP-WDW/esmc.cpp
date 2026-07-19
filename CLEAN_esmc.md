# CLEAN-esmc: Enzyme Function (EC number) Prediction on top of `esmc.cpp`

> **One-line goal.** Re-implement [CLEAN](https://github.com/tttianhao/CLEAN/tree/v1.0.0)
> (*Contrastive Learning enabled Enzyme ANnotation*, Yu et al., *Science* 2023) but
> swap its frozen **ESM-1b** protein encoder for **ESM-C 300M served by
> [`esmc.cpp`](./README.md)**, so that the whole inference path can run locally on an
> Apple-Silicon Mac (Metal/CPU, no PyTorch, no CUDA).

This document is the engineering specification and execution plan. It is written so a
single engineer **or an autonomous agent loop** can build the project end-to-end. Every
milestone has concrete tasks, deliverables, and **machine-checkable exit criteria**.

---

## 0. TL;DR for the impatient

CLEAN is conceptually small once you strip away the ESM machinery:

1. For each protein, get **one fixed-length vector** = mean-pooled embedding from a frozen
   protein language model. CLEAN uses ESM-1b (1280-d); **we use ESM-C 300M via `esmc.cpp`
   (960-d)**.
2. Train a **tiny 3-layer MLP "projection head"** (`LayerNormNet`) with a **contrastive
   loss** so that proteins sharing an Enzyme Commission (EC) number land close together and
   proteins with different EC numbers land far apart.
3. At inference: project query proteins + training proteins through the head, compute each
   EC number's **cluster center** (mean of its members), and call EC numbers for a query by
   its **distance** to those centers, using either **max-separation** (deterministic, no
   hyperparameters) or **p-value** (background-distribution thresholding).

The encoder (the expensive part) is frozen. The only thing we *train* is the MLP head,
which is ~3M parameters and trains in minutes-to-hours on CPU/GPU. So the plan is mostly:
**(a) build a fast batch embedding extractor on top of `esmc.cpp`, (b) port CLEAN's
training/inference Python to consume 960-d ESM-C embeddings, (c) optionally re-implement the
head's forward pass natively in `esmc.cpp` (ggml) so inference is a single Mac binary.**

---

## 1. How the original CLEAN works (faithful reference)

This section is the ground truth we are porting. Line references are to CLEAN `v1.0.0`.

### 1.1 Data schema
- Training/test data are **tab-separated** CSVs with header `Entry\tEC number\tSequence`.
  - `Entry`: UniProt accession (used as the embedding cache key).
  - `EC number`: one or more EC numbers separated by `;` (multi-label, e.g. `1.1.1.1;1.1.1.2`).
  - `Sequence`: amino-acid string.
- `get_ec_id_dict()` builds two maps:
  - `id_ec`: `entry -> [ec, ...]`
  - `ec_id`: `ec -> {entry, ...}` (set of members per EC).
- Splits are by sequence-identity clustering of Swiss-Prot: `split10`, `split30`, `split50`,
  `split70`, `split100` (100% ≈ full reviewed Swiss-Prot, ~220–242k sequences). Smaller
  splits = fewer/representative sequences = faster iteration.
- Held-out test sets used in the paper: **`new-392`** (392 enzymes, 177 EC) and **`price-149`**
  (a hard, experimentally-characterized set).

### 1.2 Encoder embeddings (the part we replace)
- CLEAN calls ESM's `extract.py` with `esm1b_t33_650M_UR50S --include mean` to produce, per
  protein, a `.pt` file containing `mean_representations[33]` = **mean over residue
  representations of the last layer = a 1280-d vector**.
- Everything downstream only ever sees that 1280-d vector. **ESM-1b is never fine-tuned.**

### 1.3 The trainable model: `LayerNormNet` (`src/CLEAN/model.py`)
```python
class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        self.fc1 = nn.Linear(1280, hidden_dim)   # <-- 1280 is the ESM-1b dim
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=drop_out)
    def forward(self, x):
        x = torch.relu(self.dropout(self.ln1(self.fc1(x))))
        x = torch.relu(self.dropout(self.ln2(self.fc2(x))))
        return self.fc3(x)
```
- Defaults: `hidden_dim=512`. `out_dim=128` for triplet-margin, `out_dim=256` for SupCon-Hard.
- **This is the only place the input dimension `1280` appears. For ESM-C 300M it becomes `960`.**

### 1.4 Losses (`src/CLEAN/losses.py`, `train-*.py`)
- **Triplet-margin** (`train-triplet.py`): `nn.TripletMarginLoss(margin=1, reduction='mean')`,
  i.e. `||z_a - z_p||_2 - ||z_a - z_n||_2 + margin`, Adam `lr=5e-4`, `batch_size=6000`.
- **SupCon-Hard** (`train-supconH.py`, `SupConHardLoss`): samples `n_pos` positives and
  `n_neg` negatives per anchor, temperature `T`. Paper config: `n_pos=9, n_neg=30, T=0.1`,
  `out_dim=256`. Better on small splits, slower.

### 1.5 Hard-negative EC mining (`src/CLEAN/dataloader.py`)
- `mine_hard_negative(dist_map, knn=10|30)`: for each EC, sort other ECs by current
  embedding-space distance and keep the `knn` nearest as candidate negatives, with sampling
  weight `1/distance` (closer = harder = sampled more). Skips self/zero-distance entries.
- `Triplet_dataset_with_mine_EC`: anchor = random member of an EC; positive = another random
  member of the same EC (`random_positive`); negative = a member of a hard-mined different EC
  (`mine_negative`). `__len__` = number of non-`-` EC classes.
- `MultiPosNeg_dataset_with_mine_EC`: same idea but returns `1 + n_pos + n_neg` stacked
  embeddings per item, for SupCon.

### 1.6 Distance maps & cluster centers (`src/CLEAN/distance_map.py`)
- `get_cluster_center(model_emb, ec_id_dict)`: average the projected embeddings of all members
  of each EC → one center vector per EC.
- `get_dist_map(...)`: full EC×EC pairwise (squared) Euclidean distance map between cluster
  centers — used to drive hard-negative mining.
- `get_dist_map_test(...)`: N_test × N_EC distance map between test queries and EC centers.
- Default distance is plain Euclidean on the **projected** embeddings (`dot=False`).

### 1.7 The training loop (`train-triplet.py`)
1. Precompute once per split: the ESM embedding matrix `<split>_esm.pkl` and the ESM-space
   distance map `<split>.pkl` (`compute_esm_distance`).
2. Build a `DataLoader` from a hard-negative mining of the *current* distance map.
3. Every `adaptive_rate=100` epochs: **recompute** the projected distance map with the current
   model and **re-mine** negatives (curriculum that gets harder as the head improves).
4. Save best (lowest-loss) checkpoint in the last 20% of epochs → `data/model/<name>.pth`.
- Recommended epochs scale with split size: 10%→2000, 30%→2500, 50%→3500, 70%→5000, 100%→7000.

### 1.8 EC calling at inference (`src/CLEAN/infer.py`, `evaluate.py`)
- Build EC cluster centers from the **training** set (projected). For pretrained 70/100 splits
  CLEAN ships precomputed centers (`70.pt`, `100.pt`).
- Project test queries; compute `eval_dist` = query × EC-center distance map.
- **`max-separation`** (`write_max_sep_choices` / `maximum_separation`): for each query, take
  the 10 nearest EC centers, find the largest "gap" in the sorted distance list, and predict
  every EC above the gap (deterministic, no hyperparameters). Usually best precision/recall.
- **`p-value`** (`write_pvalue_choices`): sample `nk_random` thousand training proteins to build
  a background distance distribution per EC; predict an EC if the query's distance ranks below
  `p_value * nk` of the background. Tunable sensitivity.
- Output CSV: `entry,EC:<ec>/<dist>,EC:<ec>/<dist>,...` (per the README example).
- Metrics (`get_eval_metrics`): weighted precision / recall / F1 + AUC over the multi-label
  binarized EC space.

### 1.9 Confidence (optional, `gmm.py`)
- Fits an ensemble of Gaussian Mixture Models over distances to produce a calibrated
  confidence per prediction. Nice-to-have, not core.

---

## 2. What changes for `esmc.cpp` — the design decisions

The translation is small but every decision below is load-bearing. **An implementing agent
must not silently deviate from these without recording the reason in `CLEAN_esmc_lab.md`.**

### D1 — Encoder: ESM-1b (1280-d) → ESM-C 300M (960-d), served by `esmc.cpp`
- ESM-C 300M hparams (confirmed in `src/esmc-arch.h`): `n_embd=960`, `n_layer=30`,
  `n_head=15`, `n_vocab=33`, `n_ctx=2048`.
- The per-protein feature becomes the **mean-pooled last-hidden-state, 960-d**. `esmc.cpp`
  already implements exactly this:
  - C API: `esmc_embed_mean(ctx, tokens, n_tokens, out)` (strips CLS/EOS, averages residues).
  - CLI: `esmc-embed --pool mean -o file.npy` (one sequence per process).
- **Only `LayerNormNet.fc1` changes: `nn.Linear(1280, …)` → `nn.Linear(960, …)`.** Make the
  input dim a constructor argument / CLI flag, default 960.
- **Optional later:** ESM-C 600M (`n_embd=1152`) if a GGUF is produced; keep the dim
  parameterized so this is a config change, not a code change.

### D2 — "Frozen encoder + trained head", and what "fine-tune using esmc.cpp" means
- The encoder is **frozen** in both CLEAN and here. We only train the projection head. This is
  the right and faithful design: it is cheap, reproducible, and what the paper does.
- Two inference deployment tracks (the user explicitly wants flexibility):
  - **Track A — PyTorch head (fastest to ship).** Embeddings produced by `esmc.cpp`; head
    trained and run in PyTorch. Mac-friendly (CPU/MPS), no CUDA required.
  - **Track B — Native head in `esmc.cpp`/ggml (the "all on Mac, no PyTorch" deliverable).**
    Export the trained head to GGUF and implement its forward pass (`Linear+LayerNorm+ReLU`)
    in ggml, so a single C++ binary does: tokenize → ESM-C embed → head → EC calling.
- **What we are NOT doing** (and why): full end-to-end gradient fine-tuning of ESM-C *inside*
  ggml/C++. ggml training support is immature and the paper never back-props into the encoder.
  If true end-to-end fine-tuning of the encoder is ever wanted, do it in PyTorch with the
  official ESM-C model — that is an explicitly **optional** stretch track (M11), not the plan's
  spine. The phrase "finetune using esmc.cpp" is realized as **Track B**: contrastive head
  trained on `esmc.cpp` embeddings and executed natively by `esmc.cpp`.

### D3 — Embedding extraction must be batched (CLEAN's biggest hidden cost)
- The stock `esmc-embed` CLI loads the model **per process** — fine for a handful of sequences,
  fatal for 240k. We must add a **batch extractor** that loads the model once and streams a
  FASTA/CSV, writing a cache. This is M2 and is the single most important new piece of infra.

### D4 — Embedding cache format
- CLEAN stores one `torch .pt` per protein (`data/esm_data/<entry>.pt`) holding
  `{'mean_representations': {33: tensor[1280]}}`. We replace this with a **simpler, language-
  neutral cache** so both PyTorch (Track A) and C++ (Track B) can read it:
  - `embeddings/<split>.f32.bin` — raw little-endian `float32`, row-major `[N, 960]`.
  - `embeddings/<split>.ids.json` — list of `N` entries, **same order** as rows.
  - `embeddings/<split>.meta.json` — `{model, n_embd, dtype, count, sha256}`.
- Provide a Python loader `load_embeddings(split) -> (np.memmap[N,960], id_to_row dict)` and a
  C++ reader for Track B. (A per-id `.npy` directory is also acceptable for small splits, but
  the packed `.bin` is the canonical format for scale.)

### D5 — Keep CLEAN's algorithms byte-for-byte faithful
- Mining, adaptive resampling cadence, max-sep gap heuristic, p-value background sampling,
  weighted precision/recall/F1/AUC — port **verbatim**. The only intended numeric differences
  vs. the paper come from (a) the encoder swap and (b) the input dim. Everything else identical
  makes regressions easy to localize.

### D6 — Tokenization & pooling parity
- Use `esmc.cpp`'s tokenizer (BOS=0, EOS=2, PAD=1, MASK=32). Mean-pool **excludes** BOS/EOS
  (matches `esmc-embed --pool mean` and ESM-1b's residue-mean convention). Verify in M3.

### D7 — Repo placement
- New project lives **inside this repo** under `clean_esmc/` (Python) + new `esmc.cpp` targets
  for Track B, reusing the existing `esmc` static lib, CMake, and `tools/` venv. Do **not**
  vendor CLEAN's licensed weights; we train our own head.

---

## 3. Target repository layout (to be created)

```
esmc.cpp/
├── CLEAN_esmc.md                  # this plan
├── CLEAN_esmc_lab.md              # running experiment log the agent appends to
├── clean_esmc/                    # Python package (Track A + training)
│   ├── __init__.py
│   ├── config.py                  # dims, paths, hyperparams (n_embd=960, ...)
│   ├── data.py                    # CSV/FASTA parsing, get_ec_id_dict (ported)
│   ├── embeddings.py              # cache reader/writer (.bin/.ids.json), loaders
│   ├── model.py                   # LayerNormNet (input dim parameterized)
│   ├── losses.py                  # SupConHardLoss (ported)
│   ├── dataloader.py              # mining + triplet/multiposneg datasets (ported)
│   ├── distance_map.py            # cluster centers + dist maps (ported)
│   ├── evaluate.py                # max-sep / p-value writers + metrics (ported)
│   ├── infer.py                   # infer_maxsep / infer_pvalue (ported)
│   ├── train_triplet.py           # training entrypoint
│   ├── train_supconh.py           # training entrypoint
│   └── export_head_gguf.py        # M10: dump trained head -> GGUF for Track B
├── tools/
│   └── extract_embeddings.py      # M2 fallback/orchestrator (calls C++ batch tool)
├── examples/
│   └── clean_embed/main.cpp       # M2: batch embedding extractor (loads model once)
│   └── clean_infer/main.cpp       # M10: native end-to-end EC caller (Track B)
├── tests/
│   ├── test_embed_parity.py       # M3: esmc.cpp mean-pool vs PyTorch ESM-C
│   ├── test_clean_units.py        # M4: mining/maxsep/pvalue unit tests vs fixtures
│   └── test_head_parity.py        # M10: ggml head == pytorch head
└── data/clean/                    # downloaded CSV splits + test sets (gitignored)
    embeddings/                    # embedding caches (gitignored)
    models/clean/                  # trained head checkpoints + exported GGUF
    results/clean/                 # prediction CSVs + metrics JSON
```

---

## 4. Data plan

| Artifact | Source | Notes |
|---|---|---|
| `split100.csv` (+ 10/30/50/70) | CLEAN repo `data/` | Tab-separated `Entry\tEC number\tSequence`. `split10` for fast loop. |
| `new.csv` (new-392) | CLEAN repo `data/` | Primary held-out test for metric parity. |
| `price.csv` (price-149) | CLEAN repo `data/` | Hard test set. |
| ESM-C 300M GGUF | `models/esmc-300m-f16.gguf` (existing) or HF `AnanyaPathak/esmc-300m-gguf` | F16 for accuracy; Q8_0 acceptable, see M3. |

- **Iteration ladder:** do everything first on **`split10`** (smallest) end-to-end, then scale
  to `split100`. Never block the loop on the full-Swiss-Prot extraction.
- **Licensing:** Swiss-Prot data + CLEAN's research-use license govern the CSVs; ESM-C weights
  are under the EvolutionaryScale Cambrian Open License (see repo `README.md`/`LICENSE`).
  Keep all downloaded data out of git (`.gitignore data/clean embeddings results models/clean`).

---

## 5. Milestones & exit criteria (the agent loop)

> **Agent loop protocol.** Work one milestone at a time, top to bottom. For each milestone:
> (1) implement the tasks; (2) run the **Exit-criteria commands**; (3) if all pass, append a
> dated entry to `CLEAN_esmc_lab.md` (what ran, numbers, file hashes) and advance; (4) if a
> check fails, fix and re-run — **do not advance on red**. Prefer small, verifiable commits.
> Each milestone is sized to be completable and checkable in isolation.

Legend: **DoD** = Definition of Done / exit criteria (machine-checkable).

---

### M0 — Scaffolding & environment
**Goal.** Project skeleton compiles/imports; data + model are reachable.
**Tasks.**
- Create `clean_esmc/` package and `data/clean`, `embeddings`, `models/clean`, `results/clean`
  dirs; add them to `.gitignore`.
- Ensure the existing build works: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j8`.
- Ensure `models/esmc-300m-f16.gguf` exists (convert per `README.md` §4 or download from HF).
- Create `CLEAN_esmc_lab.md` with a header.
**DoD.**
- `./build/bin/esmc-embed -m models/esmc-300m-f16.gguf -s ACDEF --pool mean` prints `mean-pool OK: 960 dims`.
- `python -c "import clean_esmc"` succeeds (empty package importable).

---

### M1 — Data acquisition & schema
**Goal.** All CSVs present and parseable into `id_ec` / `ec_id`.
**Tasks.**
- Download `split10.csv`, `split100.csv`, `new.csv`, `price.csv` into `data/clean/`.
- Implement `clean_esmc/data.py::get_ec_id_dict` (verbatim port, tab-delimited; multi-label by `;`).
- Add `csv_to_fasta` and `fasta_to_csv` helpers (ported).
**DoD.**
- `python -m clean_esmc.data --stats data/clean/split10.csv` prints `#entries`, `#unique_ec`,
  and `% multi-label`, with `#entries > 0` and EC strings matching the `d.d.d.d` shape.
- Round-trip: `csv_to_fasta` then re-parse yields identical entry set.

---

### M2 — Batch embedding extractor on `esmc.cpp` (critical infra)
**Goal.** Turn a FASTA/CSV into a packed 960-d embedding cache, loading the model **once**.
**Tasks.**
- Add C++ target `examples/clean_embed/main.cpp` (CMake `add_executable(clean-embed ...)`,
  link `esmc`). It:
  - parses a FASTA (or the CSV's `Entry`/`Sequence` columns),
  - tokenizes each sequence, runs `esmc_embed_mean` (or `esmc_embed_batch` + manual mean for
    speed), and **streams** rows to `<out>.f32.bin` while writing `<out>.ids.json`,
  - flags: `-m MODEL.gguf --in FILE --out PREFIX [--batch N] [--no-metal] [--max-len L]`,
  - skips/records sequences longer than `n_ctx` (2048) or empties; logs progress every 1k.
- Add `tools/extract_embeddings.py` orchestrator (chunking, resume-on-restart by checking the
  ids already in the cache, sharding long sequences last).
**DoD.**
- `./build/bin/clean-embed -m models/esmc-300m-f16.gguf --in data/clean/split10.csv --out embeddings/split10`
  produces `split10.f32.bin`, `split10.ids.json`, `split10.meta.json` with
  `count == #entries` and `filesize == count*960*4` bytes.
- Re-running is idempotent (resume detects a complete cache and exits fast).
- Throughput logged; a 1k-sequence subset finishes (sanity: > 5 seq/s on Metal for medium).

---

### M3 — Embedding correctness & pooling parity
**Goal.** Prove the `esmc.cpp` mean embedding matches the PyTorch ESM-C reference, so any
downstream metric gap is attributable to modeling, not a numerical bug.
**Tasks.**
- `tests/test_embed_parity.py`: for ~20 sequences, compare `clean-embed` rows against
  mean-pooled embeddings from the official PyTorch ESM-C 300M (reuse `tests/ref_forward.py`
  pattern from the existing repo) — exclude BOS/EOS in both.
- Record per-sequence cosine and L2; pick the cache dtype (start **F16**).
**DoD.**
- Mean per-sequence cosine ≥ **0.999** (F16) vs PyTorch ESM-C reference; max mean-pool L2 small
  (same order as `README.md` correctness table). If using Q8_0, cosine ≥ 0.999 still expected.
- Decision recorded in lab log: which GGUF precision is the canonical encoder.

---

### M4 — Port CLEAN core library to 960-d (PyTorch, Track A)
**Goal.** Faithful, unit-tested port of model/losses/mining/distance/eval.
**Tasks.**
- `model.py`: `LayerNormNet(in_dim=960, hidden_dim=512, out_dim)` — only `fc1` input changed.
- `losses.py`: `SupConHardLoss` (verbatim).
- `dataloader.py`: `mine_hard_negative`, `mine_negative`, `random_positive`,
  `Triplet_dataset_with_mine_EC`, `MultiPosNeg_dataset_with_mine_EC` — adapted to read the new
  cache (via `embeddings.py`) instead of per-id `.pt`.
- `distance_map.py`: `get_cluster_center`, `get_dist_map`, `get_dist_map_test`,
  `get_random_nk_dist_map` (verbatim).
- `evaluate.py`: `maximum_separation`, `write_max_sep_choices`, `write_pvalue_choices`,
  `random_nk_model`, `get_eval_metrics`, label helpers (verbatim).
- `tests/test_clean_units.py`: golden-fixture tests for `maximum_separation` (known
  distance lists → known cut index) and `mine_hard_negative` (toy dist map → expected
  neighbors/weights).
**DoD.**
- `pytest tests/test_clean_units.py` green: `maximum_separation` matches hand-computed cuts on
  ≥3 fixtures; mining returns the expected nearest ECs with `1/dist` weights summing to 1.
- `LayerNormNet(in_dim=960)` forward on a `[B,960]` tensor returns `[B,out_dim]` without shape
  errors for `out_dim ∈ {128,256}`.

---

### M5 — Precompute distance maps & cluster centers
**Goal.** One-time per-split artifacts that drive mining and inference.
**Tasks.**
- `compute_esm_distance(split)` equivalent: load cache, build EC-space ESM distance map +
  embedding matrix, persist to `embeddings/<split>.distmap.pkl` and reuse the packed `.bin`.
- Build & cache **EC cluster centers** for the training split (projected later at infer time;
  raw ESM centers now for mining bootstrap).
**DoD.**
- `python -m clean_esmc.distance_map --build split10` writes the dist-map artifact; its key set
  equals the unique non-`-` EC set; matrix is square `N_EC × N_EC` and symmetric (diag ~0).

---

### M6 — Train the triplet head (split10/30) to convergence
**Goal.** A trained head that actually separates EC classes.
**Tasks.**
- `train_triplet.py`: Adam `lr=5e-4`, `TripletMarginLoss(margin=1)`, `batch_size=6000`,
  adaptive re-mining every 100 epochs, best-checkpoint-in-last-20% logic (verbatim loop).
- Start `split10`, `--epoch 2000` (paper schedule). Log train loss per epoch.
- Save `models/clean/split10_triplet.pth` (+ exported center cache).
**DoD.**
- Training completes; **final smoothed train loss < initial loss by ≥ 50%** and is below the
  `margin=1` floor region (loss trending toward < margin), i.e. the head learns non-trivially.
- A quick sanity probe: mean intra-EC distance < mean inter-EC distance on a held-out batch
  (printed by a `--diagnose` flag).

---

### M7 — Inference + evaluation parity (max-sep & p-value)
**Goal.** Reproduce CLEAN-style EC-calling metrics with the ESM-C encoder.
**Tasks.**
- `infer.py`: `infer_maxsep(train, test, model_name)` and `infer_pvalue(...)` (verbatim flow:
  project train+test, build `eval_dist`, write choices, compute metrics).
- Run on `new-392` and `price-149` using the `split10` (and later `split100`) head.
- Emit `results/clean/<test>_maxsep.csv`, `<test>_pvalue.csv`, and a `*_metrics.json`.
**DoD.**
- `python -m clean_esmc.infer --train split10 --test new --algo maxsep --report` prints
  `precision/recall/F1/AUC` and writes the CSV in the exact CLEAN format
  (`entry,EC:<ec>/<dist>,...`).
- **Sanity bar (not paper-equality):** on `new-392` with a `split10` head, F1 is clearly
  non-trivial (target **F1 ≥ 0.30** as a smoke threshold; the split100 head in M9 is where we
  chase paper-grade numbers). Record numbers for both algos in the lab log.

---

### M8 — SupCon-Hard training variant
**Goal.** Second loss, typically stronger on small splits.
**Tasks.**
- `train_supconh.py`: `MultiPosNeg_dataset_with_mine_EC` (`n_pos=9, n_neg=30`), `SupConHardLoss`
  with `T=0.1`, `out_dim=256`, ~25% fewer epochs than triplet. Ensure `infer.py` can load a
  256-d head (parameterize `out_dim` everywhere; no hard-coded 128).
**DoD.**
- `split10_supconH` trains and evaluates via M7's path; metrics logged. `out_dim` is read from
  the checkpoint/meta, not hard-coded (verified by loading both 128-d and 256-d heads).

---

### M9 — Scale to `split100` (full Swiss-Prot)
**Goal.** Production-grade head and the headline metrics.
**Tasks.**
- Extract `split100` embeddings (M2 at scale; budget hours, use resume + Metal). Verify cache
  `count` vs CSV.
- Train `split100_triplet` (`--epoch 7000`) and optionally `split100_supconH`.
- Evaluate on `new-392` + `price-149`, both algos.
**DoD.**
- Cache integrity: `count == #entries(split100)`, `meta.sha256` recorded.
- Metrics table in lab log for {triplet, supconH} × {maxsep, pvalue} on both test sets.
- **Stretch target:** approach the paper's `new-392` ballpark (paper ESM-1b max-sep ≈
  P 0.60 / R 0.48 / F1 0.50). The ESM-C 300M encoder may land near or above this; **document the
  delta** rather than forcing equality (different encoder, different dim).

---

### M10 — Native Mac inference in `esmc.cpp` (Track B, the "all-C++" deliverable)
**Goal.** A single Mac binary: sequence(s) → EC predictions, no PyTorch at inference.
**Tasks.**
- `clean_esmc/export_head_gguf.py`: serialize the trained `LayerNormNet` (fc1/ln1/fc2/ln2/fc3
  weights+biases, `in_dim`, `hidden_dim`, `out_dim`, loss-type) into a small **GGUF** file, plus
  export the **EC cluster-center matrix** (`[N_EC, out_dim]` + EC label list) to a companion
  `.bin`/`.json`.
- `examples/clean_infer/main.cpp` (`add_executable(clean-infer ...)`): load ESM-C GGUF + head
  GGUF; for each input sequence: tokenize → `esmc_embed_mean` (960) → head forward in ggml
  (`mul_mat` + add bias + LayerNorm + ReLU ×2 + final linear) → distances to cluster centers →
  `max-separation` (port `maximum_separation` to C++) → print/write CLEAN-format CSV.
- `tests/test_head_parity.py`: assert ggml head output == PyTorch head output for the same
  input embeddings.
**DoD.**
- `tests/test_head_parity.py`: max abs diff between ggml and PyTorch head outputs < **1e-3**
  (F16) across ≥20 random inputs.
- `./build/bin/clean-infer -m models/esmc-300m-f16.gguf --head models/clean/split100_triplet.gguf -s "<AA seq>"`
  prints the same top-EC prediction as the Track-A PyTorch path for the same sequence (≥ 95%
  agreement on a 50-sequence spot check; identical max-sep cut on the agreed ones).

---

### M11 — Packaging, confidence, and optional end-to-end fine-tune (stretch)
**Goal.** Make it usable and document honestly.
**Tasks.**
- CLI ergonomics: `clean-infer --in queries.fasta --algo maxsep --out results.csv`; a thin
  Python wrapper for Track A parity.
- (Optional) Port `gmm.py` confidence ensemble over distances.
- (Optional) **True end-to-end fine-tune** of ESM-C in PyTorch (official model) as a separate
  experiment — clearly labeled optional, never on the core path. Document compute cost.
- Write a user-facing `clean_esmc/README.md` (install, extract, train, infer, benchmark) and a
  results table; cross-link from the main repo `README.md`.
**DoD.**
- A fresh-clone runbook reproduces M7 numbers on `split10` end-to-end from documented commands.
- `clean-infer` runs on a FASTA and writes a valid results CSV; README commands verified.

---

## 6. Concrete command walkthrough (the happy path, split10)

```bash
# Build (existing toolchain) + new targets
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8                      # esmc, esmc-embed, clean-embed, clean-infer

# Python env (reuse existing tools venv)
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt torch numpy scikit-learn

# 1) Data
#   download split10.csv / new.csv / price.csv into data/clean/

# 2) Embeddings (model loaded once)
./build/bin/clean-embed -m models/esmc-300m-f16.gguf --in data/clean/split10.csv --out embeddings/split10
./build/bin/clean-embed -m models/esmc-300m-f16.gguf --in data/clean/new.csv    --out embeddings/new

# 3) Distance map (mining bootstrap)
.venv/bin/python -m clean_esmc.distance_map --build split10

# 4) Train head
.venv/bin/python -m clean_esmc.train_triplet --training_data split10 --model_name split10_triplet --epoch 2000

# 5) Infer + evaluate
.venv/bin/python -m clean_esmc.infer --train split10 --test new --algo maxsep --report
.venv/bin/python -m clean_esmc.infer --train split10 --test new --algo pvalue --p_value 1e-5 --nk_random 20 --report

# 6) (Track B) Export head -> GGUF and run native
.venv/bin/python -m clean_esmc.export_head_gguf --ckpt models/clean/split10_triplet.pth --train split10 \
    --out models/clean/split10_triplet.gguf
./build/bin/clean-infer -m models/esmc-300m-f16.gguf --head models/clean/split10_triplet.gguf \
    --in data/clean/new.fasta --algo maxsep --out results/clean/new_maxsep_cpp.csv
```

---

## 7. Risks, gotchas, and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Bulk extraction is slow** (240k seqs) | M9 stalls | Batch C++ tool (M2), Metal, resume-on-restart, run long sequences last; start on split10. |
| **Dim mismatch leaks** (1280 hard-coded) | silent wrong shapes | Single `in_dim` config; grep for `1280`; assert in `LayerNormNet`. |
| **Pooling convention drift** (BOS/EOS) | metric gap vs paper | M3 parity test pins it; both paths exclude specials. |
| **Encoder-swap lowers ceiling** | F1 below paper | This is expected/allowed; document the delta. ESM-C 300M vs ESM-1b 650M is a different model — the deliverable is the *pipeline*, not beating the paper. |
| **Quantization hurts EC calling** | recall drops | Default F16 encoder; treat Q8_0/Q4 as a measured ablation, not default. |
| **ggml head numerical drift** (Track B) | predictions differ | M10 parity test < 1e-3; use F32 for the head matmuls if needed (tiny cost). |
| **Multi-label EC + `-` placeholders** | mining/eval bugs | Port `'-' not in ec` filters and `;` splitting verbatim; unit-test on fixtures. |
| **Determinism** | flaky metrics | Port `seed_everything`; fix seeds in train+infer; log seeds. |
| **License** | distribution issues | Don't commit Swiss-Prot CSVs or ESM weights; document licenses. |

---

## 8. Definition of "done" for the whole project
1. `clean-embed` produces a verified 960-d cache for split10 **and** split100.
2. M3 parity passes (encoder is numerically trustworthy).
3. Triplet **and** SupCon heads train and evaluate on `new-392` + `price-149` with logged
   precision/recall/F1/AUC for max-sep and p-value.
4. **Track B**: a single `clean-infer` Mac binary reproduces Track-A predictions within the M10
   agreement bar — i.e. **EC numbers predicted end-to-end on-device with no PyTorch**.
5. Reproducible runbook + results table committed; `CLEAN_esmc_lab.md` contains the full,
   dated experiment trail.

---

## 9. Appendix — file-by-file porting map (CLEAN → clean_esmc)

| CLEAN file | clean_esmc file | Change |
|---|---|---|
| `src/CLEAN/model.py` | `model.py` | `Linear(1280,…)` → `Linear(in_dim=960,…)`; keep `LayerNormNet`. |
| `src/CLEAN/losses.py` | `losses.py` | verbatim (`SupConHardLoss`). |
| `src/CLEAN/dataloader.py` | `dataloader.py` | mining verbatim; datasets read packed cache via `embeddings.py` (not per-id `.pt`). |
| `src/CLEAN/distance_map.py` | `distance_map.py` | verbatim. |
| `src/CLEAN/evaluate.py` | `evaluate.py` | verbatim (max-sep, p-value, metrics). |
| `src/CLEAN/infer.py` | `infer.py` | drop ESM-1b loaders; load packed cache; `out_dim` parameterized; remove hard-coded pretrained `.pt`. |
| `src/CLEAN/utils.py` | `data.py` + `embeddings.py` | `get_ec_id_dict`, `csv_to_fasta` ported; `retrive_esm1b_embedding` → `clean-embed` (C++); `load_esm`/`esm_embedding` → cache loaders. |
| `train-triplet.py` | `train_triplet.py` | same loop (adaptive re-mining, best-ckpt), new data layer. |
| `train-supconH.py` | `train_supconh.py` | same loop; `n_pos=9,n_neg=30,T=0.1,out_dim=256`. |
| `gmm.py` | `gmm.py` (optional) | confidence ensemble. |
| — (new) | `examples/clean_embed/main.cpp` | **batch ESM-C mean extractor** (M2). |
| — (new) | `examples/clean_infer/main.cpp` | **native end-to-end EC caller** (M10, Track B). |
| — (new) | `export_head_gguf.py` | dump head + EC centers to GGUF (M10). |

---

*Encoder served by [`esmc.cpp`](./README.md) (ESM-C 300M, 960-d). Algorithm faithful to
[CLEAN v1.0.0](https://github.com/tttianhao/CLEAN/tree/v1.0.0). Train the head; freeze the
encoder; run it all on a Mac.*
