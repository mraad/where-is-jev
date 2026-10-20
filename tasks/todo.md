# Documentation figures and HTML project overview

Figures come from finished outputs only (no GPU): Qwen3-VL 32B gate, SAM 3 "round storage tanks", mask suppression, evaluation inside `Extent`. Example tiles are picked by fixed rules stated in the captions, not by hand.

- [x] `figures.py`: gate tiles over the extent, detections TP/FP with missed labels, example tiles (raw masks vs kept), missed tanks → `docs/images/`
- [x] `docs/index.html`: plain HTML project description in the spirit of ASD-STE100, medium images, technical tables
- [x] README: figures in the SAM 3 section, link to `docs/index.html`; AGENTS.md module list
- [x] Verify: image sizes, every HTML image link resolves, HTML parses; tests pass

# SAM 3 "round storage tanks" after the Qwen3-VL 4B gate (512 px tiles)

Same pipeline and settings as the Qwen 32B run; only the gate changes. Reuse the completed Qwen 4B 512 px decisions (gc1 `results/tiles_gc1/512/qwen4b`), no re-run.

- [x] gc1 tmux: `patch_decisions.py gate` → `results/sam3_gate512_qwen4b`; `run_gc1.sh 32 results/sam3_round_qwen4b_512 … "round storage tanks"` (4 GPUs, mask suppression)
- [x] Copy back; `evaluate.py` → `results/sam3_round_qwen4b_validation`
- [x] GeoPackage `tank_detections_sam3_round_qwen4b_512.gpkg` → ArcGIS Tanks folder; verify
- [x] README: Qwen 4B vs 32B gate comparison

# Mask IoS suppression for the SAM 3 merge

Replace box IoU 0.5 NMS with greedy suppression on mask polygons: intersection over the smaller area > 0.5 (fixed a priori, not tuned), masks clear of their tile edge ranked before clipped ones, then score. Re-merge the existing worker outputs; no GPU re-run.

- [x] Commit the box-NMS gate work first (8e72d5e)
- [x] `merge_shards.suppress()`; `detect.py` uses it; regression test (clipped fragment with higher score loses to the whole mask)
- [x] gc1: install pinned `shapely==2.2.0` (in the lockfile, missing from gc1 `.venv`); sync code; re-merge in tmux into a fresh folder; same result locally
- [x] Re-evaluate with `evaluate.py`; label-free residual-overlap check
- [x] New GeoPackage beside the box-NMS one (kept, labeled superseded)
- [x] README, `CLAUDE.md`; commit

# SAM 3 "round storage tanks" after Qwen3-VL 32B gate (512 px tiles)

Reuse the completed Qwen 32B 512 px decisions (gc1 `results/tiles_gc1/512/qwen32b`): 265 of 736 tiles are contains/partial.

- [x] Git: initialized, baseline commit c0ae70c on `main`, branch `sam3-round-tanks`
- [x] `patch_decisions.py gate`: screen-format folder (`summary.json`, `patches.csv`, `patches.geojson`, overview) routing contains/partial tiles; unit test
- [x] `run_gc1.sh`: optional screen-folder and prompt arguments (defaults unchanged)
- [x] gc1 tmux: SAM 3, prompt "round storage tanks", threshold 0.3, BF16, 4 GPUs, then global NMS merge (`merge_shards.py`, box IoU 0.5 kept per user)
- [x] Evaluate merged detections against `Tanks` inside `Extent` with `evaluate.py`
- [x] GeoPackage: detections, gate tiles, evaluated predictions/labels → ArcGIS Tanks folder; verify
- [x] Docs: README section, `CLAUDE.md`, `AGENTS.md` module list (stale)
