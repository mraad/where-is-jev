"""Compare the four frozen model runs on the same extent-only reference patches."""

from collections import Counter
import json
from pathlib import Path

from patch_decisions import LABELS, collect

RUNS = {
    "qwen4b": "results/qwen50_benchmark/gpu-4",
    "nemotron12b": "results/nemotron50_gpu2",
    "qwen32b": "results/qwen32_gpu4",
    "nemotron30b": "results/omni30_gpu2",
}


def metrics(truth, predictions):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Expected equal, nonempty reference and prediction lists")
    matrix = [[0] * 3 for _ in LABELS]
    for actual, predicted in zip(truth, predictions):
        matrix[LABELS.index(actual)][LABELS.index(predicted)] += 1
    tp = sum(matrix[i][j] for i in (0, 1) for j in (0, 1))
    fp = matrix[2][0] + matrix[2][1]
    fn = matrix[0][2] + matrix[1][2]
    return dict(
        accuracy=sum(matrix[i][i] for i in range(3)) / len(truth),
        confusion_matrix_truth_rows_predicted_columns=matrix,
        presence_precision=tp/(tp+fp) if tp+fp else None,
        presence_recall=tp/(tp+fn) if tp+fn else None,
        presence_tp=tp, presence_fp=fp, presence_fn=fn,
    )


def main():
    reference = json.loads(Path("results/backbone_comparison.json").read_text())
    rows = reference["rows"]
    assert len(rows) == len({r["id"] for r in rows}) == 144
    report = dict(scope=reference["scope"], label_order=LABELS,
                  evaluation_patches=len(rows),
                  reference_counts=dict(Counter(r["truth"] for r in rows)), models={})
    for name, directory in RUNS.items():
        root = Path(directory)
        _, predictions, workers = collect(Path("results/patches50"), root)
        result = metrics([r["truth"] for r in rows],
                         [predictions[r["id"]]["decision"] for r in rows])
        result.update(
            model=workers[0]["model"], revision=workers[0]["revision"],
            all_patch_counts=dict(Counter(r["decision"] for r in predictions.values())),
            timing=json.loads((root / "timing.json").read_text()),
            longest_worker_load_seconds=max(w["load_seconds"] for w in workers),
            longest_worker_pipeline_seconds=max(w["pipeline_seconds"] for w in workers),
            batch_size=workers[0]["batch_size"],
        )
        report["models"][name] = result
        for row in rows:
            row[name] = predictions[row["id"]]["decision"]
    report["rows"] = rows
    with Path("results/larger_backbone_comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
