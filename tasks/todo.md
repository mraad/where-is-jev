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
