# Where Is Jev?

## SAM 3 "round storage tanks" after the Qwen3-VL 32B gate (2026-10-20)

Two stages on `Tanks.tif`, 512-pixel tiles with 50% overlap (736 tiles):

1. **Gate.** The completed Qwen3-VL 32B decisions from the multi-size run below
   (not re-run) route every tile decided *contains* or *partial* to the
   detector: **265 of 736 tiles** (235 contains, 30 partial).
   `patch_decisions.py gate` writes them as a screen-format folder, so the
   existing `detect.py` and `merge_shards.py` run unchanged. Its `margin`
   column is Qwen's weight(A) + weight(B) − weight(C), uncalibrated.
2. **Detect and merge.** `run_gc1.sh` ran SAM 3 (`facebook/sam3` revision
   `3c879f39…`, BF16, detection threshold 0.3, mask threshold 0.5, batch 32)
   with prompt **"round storage tanks"** as four independent replicas on gc1,
   one per GPU (66–67 tiles each; all four GPUs reached 100% utilization).
   `merge_shards.py` checked that every routed tile was processed exactly once,
   then applied global box NMS at IoU 0.5 to all **850 raw instances**, keeping
   **364**.

| Timing, gc1 (4 × RTX PRO 6000 Blackwell) | Seconds |
|---|---:|
| Qwen3-VL 32B decisions, 736 tiles (earlier run, startup included) | 31.2 |
| Gate: routing, CSV/GeoJSON, overview images | 1.2 |
| SAM 3 workers + merge, startup through exit | 15 |
| Longest SAM 3 worker: inference and patch outputs / model load | 7.29 / 1.20 |

Single run; SAM 3 timing has one-second resolution from the launcher.

Instance evaluation with `evaluate.py` against the 97 `Tanks` polygons uses
**only labels and detections whose centroids lie inside `Extent`**: 85
labels and 327 detections; 12 labels and 37 detections outside are ignored.
Precision assumes every tank inside the extent is labeled. Nothing was tuned
against the labels.

| Match rule | Matched tanks | Missed | Unmatched detections | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Any positive-area overlap | 85 | 0 | 242 | 26.0% | 100% | 41.3% |
| Polygon IoU ≥ 0.50 | 81 | 4 | 246 | 24.8% | 95.3% | 39.3% |

Every labeled tank inside the extent overlaps a detection; the matched IoU
median is 0.763 and labels 23, 83, 101 and 106 fail the 0.50 shape test.
Precision is low for two reasons. **Box NMS leaves duplicates**: among the 364
kept detections, 111 pairs overlap by more than half of the smaller polygon
(96 pairs by more than 80%), involving 161 detections, 100 of them clipped at a
tile edge. With 50% overlap, each tank appears in up to four tiles, and a
clipped fragment's box rarely reaches IoU 0.5 with the full tank's box. 87 of
the 246 unmatched detections overlap a labeled tank. The other 159 overlap no
label: other round structures or unlabeled tanks, not individually reviewed.
Mask-aware suppression (intersection over the smaller polygon, preferring
predictions that do not touch a tile edge) is the next step; it was not
applied here.

Output for ArcGIS Pro:
`~/Documents/ArcGIS/Projects/Tanks/tank_detections_sam3_round_qwen32b_512.gpkg`
(EPSG:4326, spatially indexed):

- `detections`: 364 instances after NMS (`score`, `patch`, `touches_patch_edge`, `pixel_box`).
- `raw_detections`: all 850 worker instances before NMS.
- `gate_tiles`: the 265 routed tiles with Qwen's `decision` and `margin`.
- `evaluated_detections`: the 327 detections inside the extent, `status` TP/FP at IoU ≥ 0.50.
- `evaluated_tanks`: the 85 labels inside the extent with `status` and `best_iou`.
- `extent`: the evaluation boundary.

Local records: `results/sam3_gate512_qwen32b/`, `results/sam3_round_qwen32b_512/`
(worker outputs, previews, logs, utilization, package freeze) and
`results/sam3_round_validation/` (summary, CSVs, `validation_overview.jpg`,
`missed_tanks.jpg`).

```bash
# gc1, inside tmux, after the multi-size Qwen32B run:
.venv/bin/python patch_decisions.py gate --manifest results/tiles_gc1/512/manifest \
  --shards results/tiles_gc1/512/qwen32b --out results/sam3_gate512_qwen32b
bash run_gc1.sh 32 results/sam3_round_qwen32b_512 results/sam3_gate512_qwen32b "round storage tanks"
# Locally, after copying both folders back:
T=~/Documents/ArcGIS/Projects/Tanks R=results/sam3_round_qwen32b_512 V=results/sam3_round_validation
.venv/bin/python evaluate.py --gdb $T/Tanks.gdb --detections $R/merged/detections.geojson --image $T/Tanks.tif --out $V
.venv/bin/python to_gpkg.py $R/tank_detections_sam3_round_qwen32b_512.gpkg \
  detections=$R/merged/detections.geojson raw_detections=$R/merged/raw_detections.geojson \
  gate_tiles=results/sam3_gate512_qwen32b/patches.geojson evaluated_detections=$V/predictions.geojson \
  evaluated_tanks=$V/ground_truth.geojson extent=$V/extent.geojson
```

## Multi-size tiles on Tanks.tif (2026-10-20)

Four decision backbones classified every tile of
`~/Documents/ArcGIS/Projects/Tanks/Tanks.tif` (8,200 × 6,087, RGB,
EPSG:3857, 0.5 m pixels, SHA-256 `1351163c…700d9`) at three tile sizes, each
with 50% overlap. Same frozen A/B/C question, checkpoints and revisions as the
comparison below; no prompt, threshold or weight was tuned on the labels.
Each model sees its tiles at full resolution, without resizing.

Reference labels come from the 97 `Tanks` polygons in `Tanks.gdb`.
**Contains** means a tank lies wholly in the tile, **partial** means a tank
overlaps it with positive area but none lies wholly inside, and **none** means
neither. **Only tiles wholly inside the `Extent` polygon are scored**; outside
tiles appear in the GeoPackage as "Not evaluated (outside extent)". Precision
assumes all tanks inside the extent are labeled. Presence counts complete and
partial as positive, and measures tiles, not tanks.

| Tile | Model | Host, GPUs | 3-class agreement | Presence precision | Presence recall | Wall time |
|---:|---|---|---:|---:|---:|---:|
| 256 | Qwen3-VL 4B | gc1, 4 | 80.3% | 78.2% | 79.4% | 15.4 s |
| 256 | Qwen3-VL 32B | gc1, 4 | **84.0%** | 73.9% | 85.4% | 49.4 s |
| 256 | Nemotron Nano VL 12B | gc5, 2 | 60.8% | 38.6% | 90.7% | 88.5 s |
| 256 | Nemotron Omni 30B-A3B | gc5, 2 | 82.9% | 61.9% | 91.8% | 76.5 s |
| 384 | Qwen3-VL 4B | gc1, 4 | 77.6% | 82.3% | 86.4% | 12.4 s |
| 384 | Qwen3-VL 32B | gc1, 4 | 79.9% | 78.1% | 92.1% | 36.2 s |
| 384 | Nemotron Nano VL 12B | gc5, 2 | 61.2% | 51.2% | 93.2% | 56.5 s |
| 384 | Nemotron Omni 30B-A3B | gc5, 2 | 76.9% | 68.0% | 94.3% | 54.4 s |
| 512 | Qwen3-VL 4B | gc1, 4 | 76.2% | **83.2%** | 86.8% | **11.3 s** |
| 512 | Qwen3-VL 32B | gc1, 4 | 77.6% | 81.0% | 91.2% | 31.2 s |
| 512 | Nemotron Nano VL 12B | gc5, 2 | 62.8% | 59.2% | 92.3% | 45.2 s |
| 512 | Nemotron Omni 30B-A3B | gc5, 2 | 73.5% | 69.9% | **94.5%** | 47.2 s |

Scored tiles (complete / partial / none): 256 px 2,296 (79 / 372 / 1,845);
384 px 999 (104 / 175 / 720); 512 px 513 (89 / 93 / 331). The four models
labeled 76–79 of 79 complete-tank tiles "contains" at 256 px, 101–103 of 104 at
384 px and all 89 at 512 px. The common error is
calling a partial-only tile "contains": Qwen4B did so for 258 of 372 at 256 px.
Nemotron12B over-predicts partial on empty tiles (555 of 1,845 at 256 px), which
drives its low precision. Smaller tiles raise the share of near-empty partial
references: at 256 px, 92 of 372 partial tiles show under 10% of any tank
(median visible fraction 31%; 60% at 512 px). Agreement at different sizes
uses different tile populations, so it is not a like-for-like accuracy ranking.

Wall time runs from launching the workers to the last worker's exit, including
model loading. Qwen ran on gc1 with 4 GPUs; Nemotron ran on gc5 with 2 GPUs,
where its vLLM environment and checkpoints already existed. Each model was
probed on 8 tiles per size first, which also absorbed first-use kernel work.
Cross-family times are not a controlled speed comparison. Single runs only.

Output for ArcGIS Pro:
`~/Documents/ArcGIS/Projects/Tanks/tank_tiles_validation.gpkg`.
Layers `tiles_256`, `tiles_384`, `tiles_512` (EPSG:3857, spatially indexed)
carry `truth`, `in_extent`, each model's decision and a `<model>_agrees` field.
Symbolize `truth`, a model field, or an `_agrees` field. The `run_summary`
table holds metrics, confusion matrices (truth rows × predicted columns),
timing and checkpoint revisions. The local copy and `summary.json` are in
`results/tiles_validation/`; worker outputs, logs, GPU utilization and package
freezes are in `results/tiles/<size>/<model>/`; probes in `results/tiles/probes/`.

```bash
.venv/bin/python -m unittest discover -v
# gc1 and gc5, inside tmux. Copy gc1's results/tiles_gc1/<size>/manifest to gc5's
# results/tiles/<size>/manifest first: all workers must share one manifest hash.
bash run_tiles.sh qwen4b qwen32b              # gc1
bash run_tiles.sh nemotron12b nemotron30b     # gc5
# Locally, after copying worker outputs into results/tiles/:
.venv/bin/python validate_tiles.py --gdb ~/Documents/ArcGIS/Projects/Tanks/Tanks.gdb \
  --image ~/Documents/ArcGIS/Projects/Tanks/Tanks.tif --root results/tiles --out results/new_tiles_validation
```

The gc1 copy of the launcher wrote to `results/tiles_gc1`; the repository
version writes to `results/tiles`, which is the only difference.

## Current workflow: standalone three-way patch decisions

### Recommended model and measured runtime

**Use Qwen3-VL-4B-Instruct for the first-stage tank-presence decision on this
image.** It achieved 89.2% presence precision, 93.0% presence recall, and 79.2%
three-category agreement. Qwen32 tied its overall agreement with lower presence
precision and a longer run. Nemotron30 improved over Nemotron12 but did not
exceed Qwen4B's overall agreement or precision on this sample.

These metrics use **only the 144 patches wholly inside the supplied extent**.
Presence combines complete and partial tanks as positive; it measures patches,
not individual tank instances. Labels are assumed complete inside the extent.
Qwen4B correctly classified 0/18 partial-only reference patches, so this
recommendation concerns presence screening; the full/partial distinction still
requires review for every tested model.

| Model | Host and GPUs | All 1,302 patches, including startup |
|---|---|---:|
| **Qwen3-VL 4B (recommended)** | gc1, 4 GPUs | **16 s** |
| Qwen3-VL 32B | gc1, 4 GPUs | 46 s |
| Nemotron Nano v2 VL 12B | gc5, 2 GPUs | 57 s |
| Nemotron 3 Nano Omni 30B-A3B | gc5, 2 GPUs | 104 s |

Qwen4B also took **38 seconds on one gc1 GPU**; four GPUs gave a measured
**2.38× wall-time speedup**, with identical labels. Cross-model timings use
different runtimes, GPU counts, batch sizes, and warmup histories, so they do
not isolate backbone speed.

The times above exclude environment setup, checkpoint downloads, preliminary
diagnostics, and GeoPackage export/copying. The two larger checkpoints each
took approximately **nine minutes to download**, concurrently on the two
hosts. A missing CUDA library on gc1 caused additional troubleshooting before
the successful Qwen32 run; its repair is documented below.

The recommended model's ArcGIS output is
`~/Documents/ArcGIS/Projects/Tanks/tank_decisions_50pct_qwen.gpkg`.
Use layer `patch_decisions` and symbolize its `decision` field. Full comparison
records are in [results/larger_backbone_comparison.json](results/larger_backbone_comparison.json).

### Completed larger-model comparison

The larger checkpoints are
[Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct),
revision `0cfaf48183f594c314753d30a4c4974bc75f3ccb`, and
[Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16),
revision `e5e9932441de940c9a62185c870ea5bcd4cd24e2`.
Qwen is dense; NVIDIA describes Omni as a roughly 31B-parameter hybrid
Mamba/Transformer mixture of experts with about 3B active parameters per token
and a CRADIO v4-H vision encoder. Total parameter count alone does not predict
runtime. Both use BF16, without quantization or task-specific training.

`run_larger_backbones.sh qwen32` uses four independent replicas on gc1;
`run_larger_backbones.sh omni30` uses two on gc5, batch size 8 per replica.
Both reuse the existing 1,302 RGB patches and frozen A/B/C question. The
Nemotron adapter explicitly sets `enable_thinking=False` and uses the Triton
MoE backend recommended by NVIDIA for RTX Pro hardware. Audio/video inputs
are disabled. The decision head is the model's existing language output head.

Downloads run separately in tmux using `download_backbone.py MODEL DIRECTORY`;
each directory records its pinned revision and download duration. Launchers
require a completed download and fresh output directory. Run logs, environment
versions, GPU utilization samples, startup-inclusive timing, and individual
worker setup/pipeline timings are saved in `results/qwen32_gpu4/` and
`results/omni30_gpu2/`. Both completed; `compare_backbones.py` compares all
four models against the same 144 reference patches wholly inside the supplied
extent. Outside patches never contribute to these precision/recall metrics.

| Model | Three-class agreement | Presence precision | Presence recall | Partial-only correct |
|---|---:|---:|---:|---:|
| Qwen3-VL 4B | 79.2% | 89.2% | 93.0% | 0/18 |
| Nemotron Nano v2 VL 12B | 66.7% | 65.0% | 94.4% | 7/18 |
| Qwen3-VL 32B | 79.2% | 80.5% | 93.0% | 5/18 |
| Nemotron 3 Nano Omni 30B-A3B | 75.7% | 78.8% | 94.4% | 6/18 |

Precision/recall here are **patch-level tank presence**, combining complete
and partial predictions as positive, not individual-tank detection metrics.
All use only the **144 patches wholly inside the sample extent**. Reference
counts are 53 complete, 18 partial-only, and 73 empty. These are agreement
measurements against the supplied polygons, conditional on complete labels
inside the extent; overlapping patches are not independent samples.

Larger Qwen recognized more complete/partial categories correctly, but made
more positive predictions on reference-empty patches: 16 versus 8 for 4B.
Its total agreement therefore tied 4B and presence precision fell. Larger
Nemotron reduced these disagreements from 36 to 18 versus 12B, with the same
presence recall. None reliably handles partial-only cases. Qwen4B remains
the most economical presence gate in this experiment; larger models do not
yet justify replacing it for that purpose.

| Larger run, all 1,302 patches | Qwen32 / gc1, 4 GPUs | Omni30 / gc5, 2 GPUs |
|---|---:|---:|
| Complete / partial / none | 338 / 58 / 906 | 315 / 87 / 900 |
| Checkpoint download, separate from inference | 530.3 s | 519.4 s |
| Successful run, startup through exit | 46 s | 104 s |
| Longest worker setup | 8.63 s | 72.38 s |
| Longest worker classification pipeline | 32.63 s | 22.63 s |

Different GPU counts, runtimes, architectures, and warmup histories prevent a
controlled backbone-speed claim. Omni's setup includes first-use kernel work;
the classification pipeline includes CPU preprocessing as well as GPU compute.
Each model processed every valid patch exactly once, and all GPUs were observed
working. Twenty unit tests passed; GeoPackages were reopened and checked for
feature count, valid categories, spatial indexing, and SQLite integrity.

Outputs in `~/Documents/ArcGIS/Projects/Tanks/`:

- `tank_decisions_50pct_qwen32b.gpkg`
- `tank_decisions_50pct_nemotron30b.gpkg`

Both contain the `patch_decisions` polygon layer; symbolize `decision`.
Detailed confusion matrices and per-patch comparisons are in
`results/larger_backbone_comparison.json`; geographic exports are in
`results/qwen32_geo/` and `results/omni30_geo/`. Neither model was trained or
tuned on the supplied ground truth. The files have not been opened in ArcGIS Pro.

On gc1, the installed `nvidia-cudnn-cu13==9.20.0.48` convolution plugin
requested `libcublasLt.so.12`, causing the initial Qwen32 run to abort.
The dynamic-linker trace reproduced this on the vision patch convolution.
Installing `nvidia-cublas-cu12==12.9.2.10` (and its
`nvidia-cuda-nvrtc-cu12==12.9.86` dependency) and adding its library directory
to the launcher's search path repaired the operation. The completed run uses
normal cuDNN convolution, not the slow native-convolution diagnostic.
Failed logs remain in `results/qwen32_gpu4_failed_cublas/` on gc1.
No checkpoint or input pixels were changed.

The current experiment classifies RGB image patches directly with a
vision-language model. It does not use the earlier Jev repositories, RemoteCLIP
screening, or SAM 3 detection stages.

- Fixed **512 × 512** model inputs, **256-pixel stride**, **50% overlap**.
- Only TIFF bands **1–3** enter the model.
- **1,302 valid patches**; 633 wholly invalid windows are omitted.
- Outer windows are padded gray to 512 × 512; their geographic footprints end
  at the raster boundary. Gray represents missing imagery.
- Exactly three output labels: **Contains a tank**, **Partially contains a tank**,
  and **Does not contain a tank**.
- “Contains” means at least one complete tank, even when partial tanks also occur.
  “Partial” means only tank fragments cut by a patch edge or missing imagery.
  Any number of tanks can be present; this is classification, not counting.

`patch_decisions.py` prepares the grid and exports a single `patch_decisions`
GeoPackage layer. `decide_qwen.py` runs the standalone
[Qwen3-VL-4B-Instruct backbone](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
on CUDA. Its existing language head scores the single-token candidates A/B/C
after the image and instruction; the highest score selects the category.
This adds no trained classification head. Candidate weights are uncalibrated.
The pinned checkpoint revision is
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

The completed gc1 run produced **325 contains / 29 partial / 948 none**.
Output: `~/Documents/ArcGIS/Projects/Tanks/tank_decisions_50pct_qwen.gpkg`.
The same predictions are in `results/qwen50_geo/patch_decisions.geojson`.

### Measured GPU use

Both Qwen runs used the same 1,302 patches, BF16, batch size 16 per worker, and
the same four-GPU gc1 host, with the runs executed sequentially:

| Measurement | One GPU | Four GPUs |
|---|---:|---:|
| Wall time, including Python/model startup | 38 s | 16 s |
| Inference pipeline, longest worker | 33.36 s | 9.59 s |
| CUDA forward time, longest worker | 24.01 s | 6.60 s |

Four GPUs gave **2.38× wall-time speedup** and identical labels on all 1,302
patches. This measures GPU-count scaling, not CPU-versus-GPU speedup. Model
vision/language computation runs on CUDA; TIFF reading, preprocessing, and
GeoPackage writing involve the CPU. One run per configuration is not a repeated
performance benchmark. Records are in `results/qwen50_benchmark/`.

### Validation and reproduction

Against the supplied polygons, Qwen agreed with the derived three-way label
on **114/144 patches wholly inside the sample extent (79.2%)**. It agreed on
49/53 complete-tank patches and 65/73 no-tank patches, but **0/18 partial-only
patches**. The full/partial distinction is therefore not reliable on this
sample. This assumes complete annotations inside the extent; an unlabeled
complete tank can invalidate a polygon-derived “partial-only” reference.
No prompt or weights were tuned against this sample.

### Nemotron backbone investigation on gc5

Tested the official
[NVIDIA-Nemotron-Nano-12B-v2-VL-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16),
revision `ca9543b126e8bf3176916d3d305ccc415f89fd4d`. NVIDIA describes a CRadioV2-H
vision encoder paired with the Nemotron Nano 12B v2 language model. The existing
language output head supplies the decision scores; no separate classifier was
trained. `decide_nemotron.py` uses greedy one-token decoding restricted to A/B/C,
with the same substantive question and class priority as Qwen. NVIDIA's
`/no_think` system setting disables free-form reasoning.

The model and candidate selection run on CUDA through **vLLM 0.32.0**, in an
isolated `.venv-nemotron` on gc5. Each of gc5's two RTX PRO 6000 Blackwell GPUs
processed 651 patches. Both GPUs were observed active; each worker reserved
about 58 GiB, including its model and inference cache. The initial 32-patch
probe incurred kernel compilation/warmup. The subsequent complete run took
**57 seconds**, including process startup; worker model/engine setup took
20.94–21.51 seconds and worker classification pipelines took 27.72–27.88 seconds.
Eager mode was used, without CUDA graph optimization.

| All 1,302 patches | Qwen / gc1 | Nemotron / gc5 |
|---|---:|---:|
| Contains a tank | 325 | 299 |
| Partially contains a tank | 29 | 350 |
| Does not contain a tank | 948 | 653 |

Models agreed on **945/1,302 patches (72.6%)**. They used identical TIFF pixels,
RGB bands, patch geometry, three class definitions, and BF16 weights/inference.
Different runtimes, backbone sizes, hosts, and GPU counts mean their wall times
are not a controlled backbone-speed comparison.

| Reference category in the 144-patch sample | Qwen correct | Nemotron correct |
|---|---:|---:|
| Complete tank, 53 patches | 49 | 52 |
| Partial tank only, 18 patches | 0 | 7 |
| No tank, 73 patches | 65 | 37 |
| Overall three-way agreement | **79.2%** | **66.7%** |

Nemotron recovered more partial-only cases but assigned “partial” to 29 of the
73 reference no-tank patches. These are disagreements with the supplied
polygons, not individually adjudicated false positives; some example patches
show small circular structures that may be unlabeled. Both models still need
review of the complete/partial distinction. This test does not support switching
to the larger backbone solely for better overall classification.

Output:
`~/Documents/ArcGIS/Projects/Tanks/tank_decisions_50pct_nemotron.gpkg`.
Both model packages contain one `patch_decisions` polygon layer in EPSG:4326;
symbolize the `decision` field for the three requested categories. Supporting
artifacts are in `results/nemotron50_geo/`, `results/nemotron50_gpu2/`, and
`results/backbone_comparison.json`. The comparison is restricted to patches
**wholly within the supplied sample extent**, preventing unlabeled areas
outside that extent from influencing their reference category. Label
completeness inside the extent remains an assumption.

```bash
.venv/bin/python -m unittest -v test_patch_decisions.py
.venv/bin/python patch_decisions.py prepare /path/to/image.tif --out results/new_patches50

# On gc1, inside tmux; performs sequential one-GPU and four-GPU runs.
bash run_decisions_gc1.sh

# Export completed workers locally.
.venv/bin/python patch_decisions.py export --manifest results/patches50 \
  --shards results/qwen50_benchmark/gpu-4 --out results/new_qwen50_geo

# On gc5, inside tmux, after setting up the documented model/runtime:
bash run_decisions_gc5.sh
```

The launcher uses the prepared project paths and refuses existing output
directories. The following sections preserve the earlier detector experiments
for provenance; their object-level evaluation is distinct from the current
patch-classification task.

## Earlier detector experiments

A local, training-free experiment: read overlapping RGB windows from a GeoTIFF,
score whether a storage tank may be present, and use SAM 3 to locate individual
instances. No annotations or fine-tuning were used for training or prompting.
User-supplied labels were subsequently used for sample-area evaluation below.
The initial run was local;
the later four-GPU run used the user's gc1 machine. The models themselves were
pretrained; “no labeling” means no new training labels from you.

## Earlier cascade design

Start with **overlapping windows → a small semantic scorer → SAM 3 → geographic
duplicate suppression**. Keep the scorer replaceable. RemoteCLIP is a useful cheap
baseline; `hr98w/jev-visual` is a closer implementation of the proposed Jev-style
yes/no interface. Neither should be trusted to discard patches until its misses
have been examined. A conventional fitted decision tree would need features and
training targets; this experiment instead uses pretrained visual models and
ordinary Python branching.

Use fixed overlapping windows first. A recursive quadtree is a later optimization:
an incorrect “no” at a coarse level hides all its descendants, precisely where
small tanks are least visible. A broad context pass can prioritize work, but
should not be the sole reason to reject a high-resolution patch.

## Research and existing implementations

Primary sources reviewed on 2026-10-20:

| Work | What already exists | Relevance and limitation |
|---|---|---|
| [hr98w/jev-visual](https://github.com/hr98w/jev-visual) | Local Qwen3.5-0.8B/MLX image decisions; existing-token scores instead of generated answers | Closest small image-only implementation. No tank-specific validation. Tested here. |
| [Official TypeSafe Jev 1.14 announcement](https://typesafe.ai/blog/jev-1-14) | Experimental still-image input in Jev 1.13/1.14, plus a single-box answer in 1.14 | Announced October 14, 2026. Another possible backend; the hosted API was not tested here. |
| [Visual Jev](https://github.com/guanxuyu-sv/Visual-Jev) and [paper](https://arxiv.org/abs/2609.25845) | Qwen3-VL-4B with a released answer-supervised adapter and shared-image question scoring | More substantial research implementation, but larger. Its benchmark gains do not establish overhead-tank accuracy. Not run here. |
| [VideoJev](https://github.com/MCG-NJU/VideoJev) | Direct candidate scoring for temporal grounding and multiple-choice video QA | Relevant principle; temporal intervals and video machinery are unnecessary for this image experiment. |
| [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) | Remote-sensing image/text encoders and zero-shot classification | Much simpler screening baseline. Scene similarity is not a reliable existential test for every tiny object. Tested here. |
| [SAHI](https://github.com/obss/sahi) and [paper](https://arxiv.org/abs/2202.06934) | Overlapping slicing, per-tile detection, coordinate restoration and merging | Already implements the tiling/detection mechanics. It does not by itself make a closed-vocabulary detector recognize a new class. |
| [ROI-Gated SAHI](https://arxiv.org/abs/2608.23923) | Lightweight region proposals before selective sliced detection | Direct precedent for the screening cascade. Its COCO128 experiment reports worse static-gating accuracy and latency; benefits depend on scene sparsity and routing. This is a limited preprint, not proof that our gate works. |
| [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) | Image grounding → segmentation, with sliced detection examples | Existing alternative using Grounding DINO plus SAM 2. No need to build a new model architecture. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Text/exemplar-prompted detection and instance segmentation | Can perform the second stage directly; an additional YOLO model is not necessary. Tested here. |
| [segment-geospatial](https://github.com/opengeos/segment-geospatial) | Geospatial wrappers around SAM-family workflows | Useful if expanding into a GIS application. Not needed for this small experiment. |

The strongest idea in the proposal is **compute allocation at the right spatial
scale**, rather than inventing a new detector.

## Specific review of hr98w/jev-visual

Source inspected at commit `4382bba455647400951429134ceb012ca155e3fe`.
The actual model tested was **`mlx-community/Qwen3.5-0.8B-4bit`**, pinned to
`da28692b5f139cb0ec58a356b437486b7dac7462`. This is a community inference
implementation, not TypeSafe's proprietary Jev weights and not VideoJev.

The [engine](https://github.com/hr98w/jev-visual/blob/main/jev_visual/engine.py)
constructs typed answers from candidate scores. Its `noul` question exposes the
normalized weight of the yes option. The default label scorer uses verified
single-token answer letters; it does not generate a JSON response. This is a
sensible fit for a narrow binary visual task.

The [adapter](https://github.com/hr98w/jev-visual/blob/main/jev_visual/adapters.py)
prefills an image/context prefix once, then branches for different questions.
It copies both attention and recurrent state, which matters for Qwen3.5's hybrid
architecture. This is **within-image sharing**, not reuse of image features
across different geographic patches. Its batch-size parameter batches question
branches, not a batch of different input images. The engine serializes requests
with a lock and should be created/called on the same thread.

The [preprocessor](https://github.com/hr98w/jev-visual/blob/main/jev_visual/preprocessing.py)
rejects images above 20 megapixels and thumbnails to at most 768 × 768.
Our 130-megapixel TIFF must therefore be windowed first. Passing a downscaled
whole-scene image and asking hundreds of region questions would save encoding
work at the expense of the object detail we need.

Its [scoring documentation](https://github.com/hr98w/jev-visual/blob/main/docs/inference.md)
explicitly says candidate probabilities are not calibrated correctness estimates.
The published shared-question throughput should not be interpreted as a
many-patch throughput benchmark. These are suitability findings from source
inspection and local probes, not a complete security audit of the project.
All 14 upstream tests passed when run from its source checkout. Testing the
installed package instead exposed a missing demo-static-files test; this did not
affect the inference API used here. The upstream demo UI is not part of this project.

Minimal upstream Python usage:

```python
from jev_visual import Request
from jev_visual.engine import Engine

engine = Engine("checkpoints/Qwen3.5-0.8B-4bit")
result = engine.judge(Request(
    image="patch_rgb.png",
    state="An RGB satellite image. Judge visible evidence.",
    questions={
        "tank": {
            "type": "noul",
            "instructions": "Does this satellite image contain a circular storage tank?",
        }
    },
))
score = result["answers"]["tank"]["noul"]
# Choose a routing threshold only after checking false negatives.
```

## Geometry of the supplied image

Source: `~/Documents/ArcGIS/Projects/Tanks/20150520_075001.tif`.

- 11,048 × 11,774 pixels, WGS84 geographic coordinates, uint8.
- Bands **1, 2, 3 only** enter the models. Band 4 is identified as alpha and is
  accessed indirectly by Rasterio's validity mask, solely for empty padding.
- Approximate centre-pixel size: **0.788 × 0.862 metres**. The affine resolution is
  in degrees, so interpreting it directly as metres would be wrong.
- Trial: **512 × 512** windows, **128-pixel overlap**, **384-pixel stride**.
  Each patch covers about 403 × 441 metres.
- Right and bottom windows are anchored to the edges; no uncovered tail strips.
  Entirely invalid windows are skipped; partially valid windows are retained.
- Invalid pixels become neutral gray for inference. A low valid fraction alone
  does not reject a patch, because a tank could occur in its small valid region.

Choose scale from `object_pixels = object_metres / metres_per_pixel`, separately
for x and y. A 20-metre tank occupies approximately 25 × 23 source pixels.
RemoteCLIP resizes a square patch to 224 × 224: in our 512-pixel trial that tank
shrinks to about 11 × 10 pixels; a 10-metre tank is only about 5 pixels across.
This explains why context-level screening can miss small isolated tanks.

Overlap has a geometric meaning. For an object bounding extent `D` plus context
margin `C` on either side, use approximately `overlap >= D + 2*C` to ensure
containment in some window away from the outer image boundary. Merely choosing
“20% overlap” is not a universal guarantee. The 128-pixel trial guarantees
containment only for extents up to 128 pixels without added context. One obvious
large tank in the diagnostic patch spans about 165 × 151 pixels, so **192-pixel
overlap is a better next trial for large tanks**, while a separate smaller-window
pass is more appropriate for tiny tanks. The experiment did not benchmark those
additional scales.

## Local implementation and results

`screen.py` reads windows with Rasterio, batches RemoteCLIP embeddings, and compares
the best tank-description similarity against the best competing-description
similarity. Alternatives include factory roofs, pipes, desert, roads, water,
ponds, and ships. This avoids relying on CLIP to understand a negated “no tank”
caption. It remains a prompt-dependent heuristic.

The margin is `max(tank cosine similarities) - max(alternative cosine similarities)`.
The exploratory yes threshold is 0. The routing threshold is -0.02, so borderline
“no” patches still reach the detector. These thresholds were not calibrated.

The complete screening run produced:

| Item | Result |
|---|---:|
| Total windows | 899 |
| Entirely invalid windows | 312 |
| Windows actually scored | 587 |
| Yes decisions | 209 |
| Borderline no decisions retained | 151 |
| Total windows routed | 360 |
| Valid windows that would be skipped | 227 / 587 = 38.7% |
| Model loading | 1.42 seconds |
| Window reading, preprocessing and batched scoring | 4.64 seconds |

This is one local MPS run, not a repeated latency benchmark. Rendering and writing
the final map/contact sheet are outside the screening timer.

Files:

- `results/screen/patches.csv`: one decision per section, including explicit
  `status=nodata` rows. A nodata “no” is not an observed negative.
- `results/screen/patches.geojson`: geographic patch footprints with the same fields.
- `results/screen/screening.jpg`: green yes patches and amber borderline patches.
- `results/screen/contact_sheet.jpg`: top, boundary, and bottom scores.
- `results/screen/summary.json`: parameters, prompts, timings, and provenance.

`detect.py` runs SAM 3 on RGB patches, clips masks to valid image pixels, maps mask
polygons to longitude/latitude, and applies global box NMS at IoU 0.5. Edge-touching
predictions are flagged. This minimal merge can retain clipped duplicates and
nested masks; mask-aware matching and seam ownership would be needed for a
trustworthy inventory.

`summarize.py` checks the completed outputs for consistent IDs, coverage, bounding
coordinates and finite scores, then creates `results/comparison.json`,
`results/detections_overview.jpg`, `results/sam3_examples.jpg`, and
`results/rejected_examples.jpg`. Its comparisons are patch-level model agreement,
not measured object recall.

The first 12-patch SAM diagnostic with `storage tank` missed obvious large tanks.
On patch `p0418`, changing the prompt to `circular storage tank` recovered three
candidates above the 0.3 threshold. More generic circular-shape prompts also
returned nested rings, so higher detection scores alone do not mean better tank
counts. The full-image baseline uses `circular storage tank`.

The completed baseline processed **all 587 valid patches** in **466.65 seconds**
including patch outputs (model loading: 0.96 seconds). It produced 941 raw
instances and **619 candidate instances after box NMS**. These are **not 619
verified tanks**: false positives, nested structures, residual seam duplicates,
and missed tanks remain. In particular, the obvious tank in `p0376` was still
missed by the circular-tank prompt.

SAM produced at least one candidate in 162 patches. The RemoteCLIP routing rule
would reject **15 of those patches**, containing 53 raw SAM predictions. Some
predictions may be false positives or also appear in retained overlapping
patches. Consequently, 15/162 is not object-level recall loss, and 53 is not a
count of missed tanks. The all-valid baseline was executed; a separate gated
SAM run was not. This is a counterfactual patch-selection comparison using the
recorded routing flags.

The Jev diagnostic found an instructive limitation:

- The detailed question returned yes on all 12 original diagnostic patches.
- The shorter question returned yes on all 15 tested patches at threshold 0.5,
  including two visually clear water patches (approximately 0.55 and 0.57).
- Clear large-tank patches scored approximately 0.96–0.98 with the short question.
  That separation may be useful for ranking, but does not establish a safe cutoff.
- One borderline patch changed decisions when answer order/wording changed.
  That check changed both order and descriptions; it is not an isolated estimate
  of option-order bias.
- `results/jev_probe.json` and `results/jev_simple.json` preserve candidate scores,
  prompt hashes and inference metrics. The two runs have different sample sets
  and are diagnostic, not a controlled accuracy comparison.

Some SAM and Jev execution overlapped on the GPU. Do not use their recorded times
as a controlled speed comparison.

## Running it

Python 3.12 was used. The main environment is pinned in `requirements-lock.txt`.
The source TIFF is always opened read-only.

```bash
uv venv --python 3.12
uv pip install -r requirements-lock.txt
.venv/bin/python -m unittest -v test_screen.py test_detect.py test_merge_shards.py test_evaluate.py

# Download the public RemoteCLIP checkpoint.
.venv/bin/python -c 'from huggingface_hub import hf_hub_download; hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", local_dir="checkpoints")'

.venv/bin/python screen.py /path/to/image.tif --out results/new_screen

# Requires a locally cached facebook/sam3 checkpoint and any applicable access.
.venv/bin/python detect.py --screen results/new_screen \
  --prompt "circular storage tank" --out results/new_detection

# Diagnostic samples include rejected patches; full baseline bypasses the gate.
.venv/bin/python detect.py --screen results/new_screen --audit \
  --prompt "circular storage tank" --out results/new_audit
.venv/bin/python detect.py --screen results/new_screen --all-valid \
  --prompt "circular storage tank" --out results/new_baseline
```

The Jev comparison uses a separate environment to preserve upstream's MLX pins.
Check out the exact upstream commit above, install its `requirements-lock.txt`
and the package there, then download its pinned checkpoint with:

```bash
.venv-jev/bin/python -m jev_visual.download --output checkpoints/Qwen3.5-0.8B-4bit
.venv-jev/bin/python jev_probe.py --chips results/audit --out results/new_jev.json
```

The probe consumes `*_rgb.png` lossless patches, not the annotated previews.
It deliberately asks two related questions per patch to inspect answer sensitivity.
The main patch scanner uses RemoteCLIP; Jev has been tested on the diagnostic
subset, not all 587 windows.

## Parallelism and the next experiment

Batch independent patches for an encoder such as RemoteCLIP. Prefetch/read the
next batch on the CPU while the GPU processes the current one if I/O becomes a
bottleneck. SAM's library supports batching; `detect.py` now accepts
`--batch-size`, `--num-shards`, and `--shard-index`. With multiple GPUs, it
distributes deterministic patch IDs across workers, then `merge_shards.py` merges
once in the original image coordinate system. On one Mac, avoid blindly spawning
a model process per patch.

For `jev-visual`, batch questions about the same patch when they are useful:
presence, circular shape, and visual ambiguity are examples. Each new patch still
needs a new visual prefill. Separate MLX engines need their own execution threads
and memory; the upstream engine is not a cross-image batch scheduler.

Let `Cg` be screening cost per patch, `Cd` detection cost, and `f` the fraction
routed. Ignoring shared I/O, the cascade helps only when `Cg + f*Cd < Cd`.
The whole-stage measured screening cost divided by 587 is about 7.9 ms per patch.
With `f=360/587`, the detector needs to cost more than about 20.4 ms per patch for
this gate to save time under that approximation. Detector runs here are much
slower than that; preservation of objects is the harder question.

The next useful comparison is direct sliced SAM 3 versus RemoteCLIP-gated SAM 3
versus Jev-gated SAM 3, with identical windows, prompts, merge policy, and a small
human-checked evaluation set. Measure **object-level recall**, not merely patch
accuracy: one missed positive patch can contain many tanks, while overlapping
patches may rescue the same object. Keep an uncertainty route and inspect a sample
of rejected patches.

[Standard COCO YOLO weights](https://docs.ultralytics.com/datasets/detect/coco/)
do not acquire a new storage-tank class from a text
question. Use an appropriate pretrained tank detector, an open-vocabulary
variant, or fine-tune later on reviewed pseudo-labels. Pseudo-labels are a useful
next step, but training on unreviewed mistakes can reinforce them.

The initial unlabeled experiment did not establish precision or recall. The
subsequent labeled sample evaluation below supplies those measurements for its
extent only; model scores remain uncalibrated and the whole-scene count unverified.

## Completed four-GPU run on gc1

On 2026-10-20, `run_gc1.sh` completed inside tmux session `where-is-jev-gpu`
in `/home/ubuntu/where-is-jev`. Four independent SAM 3 processes used all four
**NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs. This is inference
parallelism across images: one model replica per GPU, no training or
inter-GPU gradient synchronization.

The source `/home/ubuntu/Tanks/20150520_075001.tif` already existed on gc1.
Its SHA-256 matched the supplied Mac image:
`ffdb14f17ceec2e0e3bd2f754e20b6f9a024e3dd8abdb4b95b3015ad0debab69`.
Only bands 1–3 entered SAM 3. The existing screening manifest supplied the
512-pixel windows with 128-pixel overlap, but **all 587 valid patches were
processed**. The screening gate and MLX Jev model were not rerun on CUDA.

The checkpoint was `facebook/sam3`, revision
`3c879f39826c281e95690f02c7821c4de09afae7`, downloaded directly on gc1.
Its `model.safetensors` SHA-256 matched the local checkpoint:
`6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a`.
Inference used BF16 autocast with float32 stored weights, prompt
`circular storage tank`, detection threshold 0.3, mask threshold 0.5, and
global box NMS IoU 0.5.

Before the full run, four simultaneous probes each processed the same 64
patches, including preprocessing and patch-output writing:

| Batch size | Seconds | Peak allocated GPU GiB |
|---|---:|---:|
| 4 | 5.625 | 5.15 |
| 8 | 5.692 | 7.12 |
| 16 | 5.583 | 11.08 |
| 32 | 5.569 | 18.98 |

Batch 32 had the lowest observed time and was used for the full run. These
single-run timings are close enough that they do not establish a significant
throughput advantage over the smaller batches.

| Full run | Result |
|---|---:|
| Patches per GPU | 147 / 147 / 147 / 146 |
| Wall time: process startup through final merge | **20 seconds** |
| Longest worker's inference and patch-output stage | 12.52 seconds |
| Raw predictions | 936 |
| Candidate instances after global NMS | **613** |

Wall time excludes environment setup, checkpoint download, preliminary probes,
and copying results back. `utilization.csv` records all four GPUs working;
all four were observed at 100% simultaneously. The merger verified every valid
patch occurred exactly once and used each worker's **raw** predictions for the
global merge, avoiding loss from merging already-filtered worker outputs.

The earlier Mac float32 baseline yielded 619 candidates. Hardware, precision,
batching, and software versions differ, so the 613 result is neither a
bit-for-bit reproduction nor evidence of an accuracy improvement. It is also
not a controlled GPU scaling benchmark: no single-GPU full-image run was made.

The isolated gc1 environment uses Python 3.13.7, PyTorch 2.13.0+cu130,
TorchVision 0.28.0+cu130, Transformers 5.20.2, Rasterio 1.5.2, and Pillow 13.0.0.
Its full package freeze is in `results/gc1_gpu/environment.txt`. The 10 tests
passed on gc1, covering window coverage, RGB handling, geographic mask offsets,
and missing/duplicate/inconsistent shard rejection.

All outputs were copied back into this workspace:

- `results/gc1_gpu/merged/detections.geojson`: candidate mask polygons in EPSG:4326.
- `results/gc1_gpu/merged/detections_overview.jpg`: full-scene candidate boxes.
- `results/gc1_gpu/merged/summary.json`: merged counts and per-patch results.
- `results/gc1_gpu/shard-{0,1,2,3}/`: raw predictions and RGB/annotated patch previews.
- `results/gc1_gpu/timing.json`, `utilization.csv`, and `gpu-*.log`: execution records.
- `results/gc1_probe/batch-*/summary.json`: preliminary batch measurements.

To inspect the completed tmux session or repeat the run using a fresh directory:

```bash
ssh -t gc1 'tmux attach -t where-is-jev-gpu'

# Inside a tmux shell on gc1:
cd /home/ubuntu/where-is-jev
bash run_gc1.sh 32 results/gc1_repeat
```

The launcher refuses an existing run directory. It assigns disjoint rows with
`rows[gpu_index::4]`, sets `CUDA_VISIBLE_DEVICES` per worker, waits for all four
to succeed, then validates and merges. Check `COMPLETE` for success or worker
logs if a run fails. Candidate counts remain unverified; false positives,
misses, nested rings, and clipped seam duplicates are still possible.

## Ground truth validation inside the supplied extent

Evaluated the saved `Extent` and `Tanks` layers in
`~/Documents/ArcGIS/Projects/Tanks/Tanks.gdb` against the **existing
613 gc1 predictions**. No detector threshold, prompt, or weights were changed
using the labels. The original FileGDB was read only.

**Scope: precision and recall use ONLY detections and ground truth inside the
saved extent polygon. Outside detections do not contribute true or false
positives, and outside labels do not contribute false negatives.**

The saved extent contains **66 of the 73 labeled tanks** and **114 detections**.
Seven labels and **499 detections outside the extent are ignored**.
Inclusion is determined by the original polygon centroid being inside or on
the extent boundary. In this run, all 114 retained prediction polygons and all
66 labels are also entirely within the extent; none requires clipping.
Retained polygons are clipped before comparisons if needed, and areas are
calculated in **UTM zone 37N (EPSG:32637)** rather than square degrees or
Web Mercator.

Matching is one-to-one: process predictions in descending confidence, assigning
each its highest-IoU eligible unmatched label. Merely touching polygon edges
does not qualify. IoU is intersection area divided by union area.

| Match rule | Matched tanks | Missed tanks | Unmatched predictions | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Any positive-area overlap | 62 | 4 | 52 | 54.4% | 93.9% | 68.9% |
| Polygon IoU ≥ 0.25 | 62 | 4 | 52 | 54.4% | 93.9% | 68.9% |
| Polygon IoU ≥ 0.50 | 61 | 5 | 53 | 53.5% | 92.4% | 67.8% |

**Precision assumes that all tanks inside the extent are labeled.** If these
are only a subset even within that extent, unmatched predictions require review
and cannot all be called false positives. These are fixed-threshold sample
metrics, not average precision or a whole-scene accuracy estimate.

The earlier intersection-based report included two predictions with only about
2% of their area inside the extent and their centroids outside. Those two are
now excluded. The current report supersedes the earlier 116-prediction metrics.

Specific findings:

- Tank OBJECTIDs **37, 38, 39, and 49** have no prediction overlap.
- OBJECTID **36** is partially segmented; its best polygon IoU is **0.354**.
  It passes the overlap test but fails the 0.50 shape-agreement test.
- The 61 matches at IoU ≥ 0.50 have median IoU **0.855**.
- Of 53 unmatched predictions at that threshold, 16 overlap a label but fail
  matching and 37 overlap no label. The previews show partial masks and nested
  segments; these should be reviewed before treating every unmatched mask as
  a separate false object.

The detector finds most labeled tanks, but the output still needs better
instance cleanup and investigation of the four clear misses. The original
box-NMS merge is not sufficient for a reliable tank inventory. No parameters
were tuned to this sample during validation.

Run again with a fresh output directory:

```bash
.venv/bin/python -m unittest -v test_evaluate.py
.venv/bin/python evaluate.py \
  --gdb ~/Documents/ArcGIS/Projects/Tanks/Tanks.gdb \
  --detections results/gc1_gpu/merged/detections.geojson \
  --image ~/Documents/ArcGIS/Projects/Tanks/20150520_075001.tif \
  --out results/new_validation
```

Final results are in **`results/validation_inside_extent/`**:

- `summary.json`: counts, metrics, explicit extent-only scope note, input prediction hash.
- `validation_overview.jpg`: green matched labels, red missed labels, cyan
  predictions, and yellow extent, using the IoU ≥ 0.50 assignment.
- `missed_tanks.jpg`: close-ups of all five labels missed at IoU ≥ 0.50.
- `predictions.geojson`, `ground_truth.geojson`, `extent.geojson`: geographic
  layers for GIS review; prediction status and label IDs preserve the matching.
- `ground_truth.csv`: one row per evaluated label, including best IoU.
- `overlaps.csv`: every positive-area prediction/label intersection.

An earlier `validation_initial/` run saw an empty extent and correctly withheld
precision; its 73-label recall result is superseded by this extent-specific
evaluation. Five evaluation tests cover reprojection axis order, clipping,
boundary selection, duplicate matching, edge-only contact, and missing-extent
metric handling, including a regression check that outside detections cannot
affect extent-only precision or recall. Implementation uses the documented
[Pyogrio layer reader](https://pyogrio.readthedocs.io/en/latest/introduction.html)
and [Shapely geometry transformation](https://shapely.readthedocs.io/en/stable/reference/shapely.transform.html).
