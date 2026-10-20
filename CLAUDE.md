# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

AGENTS.md lists the modules; the notes below cover how they fit together. `README.md` is the experiment log (methods, pinned revisions, results, caveats); update it when results change, and keep superseded results labeled rather than deleted.

## Commands

```bash
.venv/bin/python -m unittest -v test_patch_decisions.PatchDecisionTests.test_fixed_half_overlap_and_complete_coverage
```

The `decide_*.py` scripts need a CUDA GPU, so they cannot run on this Mac. The `.sh` launchers run in tmux on gc1 (4 GPUs) or gc5 (2 GPUs) with hardcoded host paths; read each script for what it checks and writes. Extra environments: `.venv-jev` (`jev_probe.py` only) and `.venv-nemotron` (gc5, vLLM); the gc1 package freeze is `results/gc1_gpu/environment.txt`.

## Current pipeline: three-way patch classification

1. `patch_decisions.py prepare --size N` (default 512): N px windows at 50% overlap → `patches.geojson` + `manifest.json`.
2. `decide_qwen.py` (transformers) / `decide_nemotron.py` (vLLM): one process per GPU via `--shard-index/--num-shards`. The decision is the highest-scoring A/B/C letter token after the frozen `QUESTION` (Qwen: one forward pass; Nemotron: one constrained greedy token); no trained head. Needs `<model>/provenance.json` from `download_backbone.py MODEL DIR`.
3. `patch_decisions.py export`: `collect()` rejects inconsistent or incomplete shards, then writes the `patch_decisions` GeoPackage layer.
4. `compare_backbones.py`: scores the runs in `RUNS` against `results/backbone_comparison.json`; refuses to overwrite its output.

Multi-size run on `Tanks.tif`: `run_tiles.sh MODEL...` (all sizes for each model, one replica per GPU of the host; Qwen on gc1, Nemotron on gc5) writes `results/tiles/<size>/<model>/`. Prepare manifests once and copy them to the other host, since `collect()` requires identical manifest hashes. `validate_tiles.py` derives reference labels from `Tanks.gdb`, scores tiles wholly inside `Extent`, and writes one GeoPackage.

Detection after the gate: `patch_decisions.py gate` writes contains/partial tiles in `screen.py`'s folder format, so `run_gc1.sh BATCH RUN_DIR GATE_DIR PROMPT` drives the unchanged `detect.py --all-valid` and `merge_shards.py`; `evaluate.py` then scores instances and `to_gpkg.py` packages the layers. `merge_shards.suppress()` (also used by `detect.py`) drops a mask overlapping a kept one by more than half the smaller area, ranking masks clear of their tile edge first; it replaced box NMS, which left clipped-fragment duplicates at 50% overlap (see README). It imports shapely, so gc1's `.venv` needs `shapely`. `figures.py` renders `docs/images/` for `README.md` and `docs/index.html`; re-render them when the featured run changes.

Coupling: `LABELS` and `QUESTION` in `patch_decisions.py` are shared by both decide scripts, `collect()` and `compare_backbones.py`. `LABELS` order maps to A/B/C. Editing `QUESTION` changes `prompt_sha256`, and re-running `prepare` changes `manifest_sha256`; either makes existing worker outputs incompatible with new ones.

In the earlier pipeline, `merge_shards.py` applies global NMS to the workers' **raw** predictions, not their filtered outputs.

## Data and reporting

- Source image: `~/Documents/ArcGIS/Projects/Tanks/20150520_075001.tif` (same file as `/home/ubuntu/Tanks/` on gc1). Ground truth: `Tanks.gdb` in that folder (`Extent`, `Tanks` layers). Deliverable `.gpkg` files are written there.
- Patch metrics are **patch-level presence** (complete + partial counted as positive) on the 144 patches wholly inside the extent; they are not tank-instance metrics. `evaluate.py` gives instance metrics for the SAM 3 detections.
- No prompt, threshold or weight is tuned against the ground truth. Cross-model timings use different hosts, GPU counts and runtimes; don't present them as controlled speed comparisons.
