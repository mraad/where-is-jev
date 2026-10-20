# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python experiment for detecting storage tanks in overhead imagery without task-specific training.

- `screen.py`: overlapping RGB windows and RemoteCLIP screening.
- `detect.py`: SAM 3 inference, batching, and GPU sharding.
- `merge_shards.py`: coverage validation and global duplicate suppression.
- `evaluate.py`: polygon matching against FileGDB ground truth within the sample extent.
- `jev_probe.py`: separate MLX/Jev diagnostic; `summarize.py`: local experiment summaries.
- `test_*.py`: standard-library unit tests beside the modules.
- `run_gc1.sh` and `benchmark_gc1.sh`: remote GPU execution helpers.
- `checkpoints/` and `results/`: ignored model weights and generated artifacts. `README.md` records research, methods, and reproducible runs.

## Build, Test, and Development Commands

There is no build step. Set up the local Python 3.12 environment:

```bash
uv venv --python 3.12
uv pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -v
```

Run screening and detection with fresh output directories:

```bash
.venv/bin/python screen.py /path/to/image.tif --out results/new_screen
.venv/bin/python detect.py --screen results/new_screen --all-valid \
  --prompt "circular storage tank" --out results/new_detection
.venv/bin/python evaluate.py --gdb /path/to/Tanks.gdb \
  --detections results/new_detection/detections.geojson --out results/new_validation
```

Model checkpoints must be available locally. For CUDA dependencies, tmux execution, and the separate Jev environment, follow `README.md`; the local lockfile is not the gc1 environment specification.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, and descriptive module names. Follow existing `argparse`, `pathlib`, and small-function patterns. No formatter or linter is configured. Keep changes focused and validate CLI inputs before writing outputs.

## Testing Guidelines

Use `unittest`; name files `test_<module>.py` and methods `test_<behavior>`. Add focused regression tests for geometry, coordinate transforms, patch coverage, matching, and shard validation. No numerical coverage target is configured. Tests should use synthetic data and avoid downloading weights or requiring GPUs.

## Commit & Pull Request Guidelines

This checkout has no Git metadata, so no historical commit convention is established. Use concise imperative subjects, such as `Exclude outside detections from evaluation`. PRs should explain behavior changes, validation commands, and relevant issues. Include preview images when geographic outputs change; document model revisions, thresholds, and precision changes affecting results.

## Data & Evaluation Rules

Keep source TIFFs and FileGDBs read-only. Pass only bands 1–3 to models. Never commit credentials, checkpoints, or generated results. Use fresh output directories. Evaluate only detections and labels whose original centroids lie inside the sample extent; explicitly document exclusions. Precision assumes complete labels within that extent.
