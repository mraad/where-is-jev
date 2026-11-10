# Candidate filtering with vision-Kev

A completed feasibility experiment on gc5, 2026-11-10. This is supervised
adaptation after SAM 3, separate from the project's training-free baseline.

**Result:** the simple linear image classifier outperformed both untuned
image-Kev and a fine-tuned Kev decision head on this spatial holdout. Do not
deploy a filter from this single-scene result.

## Data and protocol

Inputs are the 262 retained detections from the full 736-tile SAM 3 retest.
Only the 234 candidates whose original centroids were inside the labeled
extent were considered. RGB crops include context around each candidate:
1.5 times its largest box dimension, at least 128 source pixels, resized
to 224 × 224. No label outlines are drawn into the inputs.

The extent's horizontal pixel span is divided into fixed geographic bands:
left 50% training, next 25% calibration, right 25% test. Any crop crossing a
band boundary or raster edge is excluded. This excluded 14 candidates.
The bands contain 105/44/71 candidates. Only 100 training examples were used:
32 with best reference IoU ≥ 0.50 and 68 with zero overlap; five ambiguous
overlaps were excluded from training. These are geometry-derived labels,
not independently reviewed human classifications.

No reference tank overlaps candidate polygons in multiple partitions.
This check and non-overlapping crop bands reduce leakage, but one image
does not test transfer to another location, acquisition date, or sensor.

The same frozen Qwen3.5 vision features feed two approaches:

- A standardized, balanced logistic regression on mean-pooled visual tokens,
  C=1, trained on the 100 clear training examples.
- Kev-0.8B with the original vision encoder reconnected to its text backbone.
  Image tokens use spatial rotary positions; the released adapter is retained.
  Only the pointer head is fine-tuned, for 40 full-batch AdamW steps at 2e-5.
  The vision encoder, language backbone and LoRA adapter stay frozen.

The untuned image-Kev is also measured. This tests head fine-tuning, not
end-to-end vision or LoRA training. The image integration follows the approach
described in [Kev issue #230](https://github.com/jaredpalmer/kev/issues/230);
that upstream example is an unbenchmarked prototype.

Each method's threshold is selected using calibration data only: the strictest
threshold retaining at least 95% of SAM's originally matched calibration
detections. Scores are not calibrated probabilities. Epoch count and
hyperparameters were fixed before reading test scores; there was no search.

## Held-out results

The test band has 13 reference tanks and 71 eligible candidate crops.
Matching uses the existing evaluator: one-to-one, original SAM confidence
order, polygon IoU ≥ 0.50. Reference tanks in the test band remain in the
denominator even when no eligible crop covers them.

| Filter | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| None | 11 | 60 | 2 | 15.5% | 84.6% | 26.2% |
| Linear vision classifier | 11 | 6 | 2 | 64.7% | 84.6% | 73.3% |
| Image-Kev, untuned | 11 | 38 | 2 | 22.4% | 84.6% | 35.5% |
| Image-Kev, tuned head | 11 | 35 | 2 | 23.9% | 84.6% | 37.3% |

All filters retained the 11 existing true positives. Head tuning gave a small
improvement over untuned Kev, but the linear probe removed substantially more
false positives. This is not evidence that larger Kev models or LoRA training
cannot help. It is evidence to collect more independent labeled imagery
before paying for a larger experiment. This test partition has now been read;
it must not become a tuning set while still being called an untouched test.

## Implementation

The entry point is [kev_filter.py](kev_filter.py), run as
`python -m experiments.kev_filter` from the repository root. It reuses
[the project evaluator](../evaluate.py); it does not alter `detect.py`,
`merge_shards.py`, or the pretrained gate.

```text
retained SAM polygons + reference evaluation + source RGB
  → context crops and spatial partitions
  → frozen Qwen visual tokens
      → mean pooling → train-only scaler → logistic regression
      → Kev text backbone + released LoRA → question/option representations
          → original pointer head
          → pointer head trained on clear training examples
  → calibration-only acceptance thresholds
  → filter test candidates → re-match reference polygons
```

| Function | Responsibility |
|---|---|
| `prepare` | Join evaluation IDs to retained detections, build RGB crops, assign bands, and derive training labels. |
| `partition` | Return a band only when the whole crop fits; otherwise mark it excluded. |
| `representations` | Load Kev and its original vision encoder, insert image tokens with spatial positions, and cache frozen representations. |
| `select_threshold` | Select the strictest cutoff preserving at least 95% of the originally matched calibration candidates. |
| `evaluate_detections` | Reuse polygon matching at IoU 0.50, retain original SAM ranking, and check for reference objects shared across partitions. |
| `run` | Fit the linear probe and Kev head, save scores and weights, then complete detection evaluation. |

Feature extraction runs on CUDA with BF16 backbones. The cached representations
and pointer head are moved to CPU, where the small head is trained. The linear
classifier also fits on CPU. The scaler is fitted only on training examples.
The original Kev head retains its released temperature when producing its
baseline scores; the tuned head uses temperature 1.0. Acceptance thresholds
are fitted separately for each method.

### Input contract

Supply the same source TIFF and FileGDB used for the SAM evaluation.
The FileGDB must contain `Tanks` and `Extent` layers. `--inputs` must contain:

```text
candidate/
  merged-retry/detections.geojson
  evaluation-retry/predictions.geojson
  evaluation-retry/extent.geojson
```

The prediction `id` must be the zero-based feature index in the retained
detections file. Do not reorder either file independently. Retained detections
must carry their original `pixel_box`; evaluated predictions need `id`,
`score`, `best_iou`, `status`, and `matched_truth_id`. The existing pipeline
produces those fields. The pilot assumes its original uint8 RGB raster and
projected reference CRS; it is not a generic dataset importer.

### Output contract

| Artifact under `--out` | Contents |
|---|---|
| `crops/<id>.png` | RGB context crops for eligible candidates. |
| `records.json` | Geographic cut positions and all candidate assignments, including exclusions. |
| `features.pt` | Mean visual features, Kev decision/option representations, and original-head logits. |
| `initial_head.pt`, `tuned_head.pt` | Pointer-head state dictionaries; not complete standalone Kev checkpoints. |
| `linear_probe.npz` | Training mean/scale, logistic coefficients, and intercept. |
| `scores.json` | Candidate IDs, labels, partitions, and all three classifier scores. |
| `report.json` | Frozen thresholds, candidate retention, re-matched detection metrics, and overlap audit. |
| `provenance.json` | Resolved model revisions, Torch version, GPU, and extraction duration. |

A prediction is accepted when its score is greater than or equal to its
method's saved threshold. The measured thresholds were 0.026407 for the
linear classifier, 0.200772 for original image-Kev, and 0.175008 for the tuned
head. These values belong to this fitted model and calibration region; they
are not defaults for a new location.

The CLI saves an experiment, not a production inference service or filtered
GeoPackage. Reject a run if evaluation fails or the cross-partition overlap
audit reports shared reference objects. A preliminary `report.json` exists
before final evaluation, so confirm that `detection_evaluation` and
`reference_objects_crossing_partitions: []` are present and the command
exited successfully.

## Runtime and artifacts

One RTX PRO 6000 Blackwell on gc5, in a separate Python 3.12 environment:
PyTorch 2.8.0+cu128, Transformers 5.20.2, PEFT 0.21.0. No existing inference
environment was modified. Frozen feature extraction for 220 eligible crops
took 11.11 seconds, excluding model loading; this combined extraction time
is not a production speed comparison between the filters.

- Kev source: local checkout commit 2074cc4.
- Kev-0.8B v1.0: bf75a6a8848ea6960ff2ed108d9ed44c2941174f.
- Qwen/Qwen3.5-0.8B-Base: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68.
- Remote workspace: /home/ubuntu/dh/kev-filter-20261110/.
- Results, frozen features and trained heads: pilot-v1/ under that workspace.
- Local archive: results/kev_filter_pilot/evidence.tar.gz (ignored).
- The archive includes scores, split records, report, provenance, model heads,
  linear coefficients/scaler and an exact environment freeze.

Run from this repository with the inspected Kev source on PYTHONPATH and
the dependencies from the recorded environment. Input files retain their
relative paths under candidate/merged-retry and candidate/evaluation-retry.

```bash
PYTHONPATH=/path/to/kev:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python -m experiments.kev_filter \
  --image /path/to/Tanks.tif --gdb /path/to/Tanks.gdb \
  --inputs /path/to/sam3-retest-inputs --out results/new_kev_filter_pilot
```

The script refuses an existing output directory. The current CLI includes
polygon evaluation, which was run as a separate completion step for pilot-v1.
It resolves the v1.0 model tag and records the full commit; for exact repeats
verify that revision against the one above.

Validation: all 28 local unit tests pass, including crop boundary exclusion
and calibration-threshold behavior; the complete GPU feature extraction,
head training and polygon evaluation finished on gc5. No production filter
was enabled and no full-backbone or LoRA training was attempted.

## Complete CLI rerun

The documented implementation was rerun end to end on gc5 on 2026-11-10,
including polygon evaluation in the same CLI invocation. Outputs are in
`/home/ubuntu/dh/kev-filter-20261110/pilot-v2/`; the launch script is
`rerun.sh` and the completion marker is `RERUN_COMPLETE`.

The full reports and detection metrics matched pilot-v1 exactly. Candidate
split records and all three classifiers' scores were byte-identical.
Model revisions, Torch version, and GPU type also matched. Frozen feature
extraction took 11.12 seconds versus 11.11 seconds on the first run.
Both focused regression tests passed in the gc5 environment.

Local evidence: `results/kev_filter_pilot/rerun-evidence.tar.gz` (ignored),
including the executed source, launch script, scores, models, and
`rerun-comparison.json`. This verifies reproducibility on the same data and
environment; it is not a new independent accuracy evaluation.
