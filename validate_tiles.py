"""Score multi-size tile decisions against FileGDB tanks inside the extent; write one GeoPackage."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from pyogrio.raw import read, write
from pyproj import CRS
import rasterio
from rasterio.windows import Window, bounds
import shapely

from compare_backbones import metrics
from evaluate import project, read_layer
from patch_decisions import LABELS, collect

MODELS = ("qwen4b", "qwen32b", "nemotron12b", "nemotron30b")
OUTSIDE = "Not evaluated (outside extent)"


def reference_labels(tiles, tanks):
    """Complete if a tank lies wholly in the tile; partial if one only overlaps it with positive area."""
    tanks = np.asarray(tanks, dtype=object)
    labels = []
    for tile in tiles:
        if shapely.covers(tile, tanks).any():
            labels.append(LABELS[0])
        elif (shapely.area(shapely.intersection(tile, tanks)) > 0).any():
            labels.append(LABELS[1])
        else:
            labels.append(LABELS[2])
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gdb", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="Directory of <size>/manifest and <size>/<model> runs")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a fresh output directory")
    image_sha = hashlib.sha256(a.image.read_bytes()).hexdigest()
    with rasterio.open(a.image) as src:
        crs, transform, shape = src.crs, src.transform, [src.height, src.width]
    tanks, tanks_crs = read_layer(a.gdb, "Tanks")
    extents, extent_crs = read_layer(a.gdb, "Extent")
    if len(extents) != 1:
        raise ValueError("Expected exactly one extent polygon")
    tanks = [g if CRS(tanks_crs) == CRS(crs) else project(g, tanks_crs, crs) for _, g in tanks]
    extent = extents[0][1] if CRS(extent_crs) == CRS(crs) else project(extents[0][1], extent_crs, crs)

    a.out.mkdir(parents=True, exist_ok=True)
    gpkg = a.out / "tank_tiles_validation.gpkg"
    runs, layers = [], {}
    for size_dir in sorted((d for d in a.root.iterdir() if (d / "manifest").is_dir()), key=lambda d: int(d.name)):
        manifest_dir = size_dir / "manifest"
        manifest = json.loads((manifest_dir / "manifest.json").read_text())
        if manifest["image_sha256"] != image_sha or manifest["shape"] != shape:
            raise ValueError(f"{manifest_dir} was prepared from a different image")
        size = manifest["patch_size"]
        features = json.loads((manifest_dir / "patches.geojson").read_text())["features"]
        rows = [f["properties"] for f in features]
        tiles = [shapely.box(*bounds(Window(r["x"], r["y"], r["width"], r["height"]), transform)) for r in rows]
        inside = [bool(extent.covers(t)) for t in tiles]
        truth = reference_labels(tiles, tanks)
        scored = [i for i, ok in enumerate(inside) if ok]
        fields = dict(patch_id=[r["id"] for r in rows], valid_fraction=[r["valid_fraction"] for r in rows],
                      in_extent=[int(ok) for ok in inside],
                      truth=[t if ok else OUTSIDE for t, ok in zip(truth, inside)])
        for model in MODELS:
            run_dir = size_dir / model
            if not (run_dir / "COMPLETE").exists():
                continue
            _, decisions, workers = collect(manifest_dir, run_dir)
            predicted = [decisions[r["id"]]["decision"] for r in rows]
            fields[model] = predicted
            fields[f"{model}_agrees"] = ["yes" if t == d else "no" if ok else OUTSIDE
                                         for t, d, ok in zip(truth, predicted, inside)]
            timing = json.loads((run_dir / "timing.json").read_text())
            result = metrics([truth[i] for i in scored], [predicted[i] for i in scored])
            runs.append(dict(tile_size=size, model=model, checkpoint=workers[0]["model"],
                             revision=workers[0]["revision"], tiles=len(rows), tiles_evaluated=len(scored),
                             accuracy=result["accuracy"], presence_precision=result["presence_precision"],
                             presence_recall=result["presence_recall"], presence_tp=result["presence_tp"],
                             presence_fp=result["presence_fp"], presence_fn=result["presence_fn"],
                             confusion_matrix=json.dumps(result["confusion_matrix_truth_rows_predicted_columns"]),
                             gpus=timing["gpus"], batch_size=workers[0]["batch_size"],
                             wall_seconds=timing["wall_seconds"],
                             longest_worker_load_seconds=max(w["load_seconds"] for w in workers),
                             longest_worker_pipeline_seconds=max(w["pipeline_seconds"] for w in workers),
                             all_tile_counts=json.dumps(Counter(predicted))))
        layer = f"tiles_{size}"
        write(gpkg, np.array([t.wkb for t in tiles], dtype=object),
              [np.array(v, dtype=object if isinstance(v[0], str) else None) for v in fields.values()],
              list(fields), layer=layer, driver="GPKG", geometry_type="Polygon", crs=crs.to_string(),
              layer_options={"SPATIAL_INDEX": "YES"})
        layers[layer] = dict(tiles=len(rows), evaluated=len(scored),
                             reference_counts=dict(Counter(truth[i] for i in scored)))
    if not runs:
        raise ValueError("No completed model runs found")
    write(gpkg, None, [np.array([r[k] for r in runs], dtype=object if isinstance(runs[0][k], str) else None)
                       for k in runs[0]], list(runs[0]), layer="run_summary", driver="GPKG")

    for layer, info in layers.items():
        _, _, geometry, _ = read(gpkg, layer=layer)
        assert len(geometry) == info["tiles"]
    assert len(read(gpkg, layer="run_summary", read_geometry=False)[3][0]) == len(runs)
    with sqlite3.connect(gpkg) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    summary = dict(image=str(a.image), image_sha256=image_sha, gdb=str(a.gdb), crs=crs.to_string(),
                   tanks=len(tanks), layers=layers, runs=runs,
                   scope="Only tiles wholly inside the Extent polygon are scored; labels assumed complete there.",
                   reference_rule="Complete: a tank polygon lies wholly in the tile (clipped to the raster). "
                                  "Partial: a tank overlaps it with positive area but none lies wholly inside. "
                                  "None: no positive-area overlap. Patch-level presence, not tank counts.")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(layers=layers, runs=[{k: r[k] for k in ("tile_size", "model", "accuracy",
          "presence_precision", "presence_recall", "wall_seconds")} for r in runs]), indent=2))


if __name__ == "__main__":
    main()
