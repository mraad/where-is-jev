"""Validate complete SAM 3 shards, then suppress duplicates across all raw predictions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely import STRtree
from shapely.geometry import shape


def suppress(objects, threshold=0.5):
    """Greedy NMS on mask polygons by intersection over the smaller area; returns kept indices.

    Masks clear of their tile edge rank before clipped ones, then by score, so a whole tank
    from one tile suppresses its clipped fragments from the overlapping tiles.
    """
    # Area ratios in EPSG:4326 match metric ones at tile scale.
    polygons = [shape(o["geometry"]) for o in objects]
    if any(p.is_empty or not p.is_valid for p in polygons):
        raise ValueError("Invalid prediction masks")
    tree = STRtree(polygons)
    kept = []
    for i in sorted(range(len(objects)), key=lambda i: (objects[i]["properties"]["touches_patch_edge"],
                                                        -objects[i]["properties"]["score"])):
        p = polygons[i]
        if not any(j in kept and p.intersection(polygons[j]).area > threshold * min(p.area, polygons[j].area)
                   for j in tree.query(p, predicate="intersects")):
            kept.append(i)
    return kept


def merge_shards(shards, screen, out, num_shards=4):
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output is not empty; choose a new --out directory")
    meta = json.loads((screen / "summary.json").read_text())
    with (screen / "patches.csv").open() as f:
        expected = [r["id"] for r in csv.DictReader(f) if r["status"] == "scored"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Screen must contain unique scored patch IDs")
    workers, objects, patch_results = [], [], []
    common_fields = ("model", "prompt", "threshold", "selection", "dtype", "image", "rgb_bands")
    for index in range(num_shards):
        directory = shards / f"shard-{index}"
        worker = json.loads((directory / "summary.json").read_text())
        if worker["shard_index"] != index or worker["num_shards"] != num_shards:
            raise ValueError(f"Incorrect shard identity: {directory}")
        if worker["selection"] != "all_valid":
            raise ValueError("Complete coverage requires all_valid selection")
        if worker["image"] != meta["source"] or worker["rgb_bands"] != [1, 2, 3]:
            raise ValueError(f"Source image or RGB bands mismatch: {directory}")
        for key in common_fields:
            if workers and worker[key] != workers[0][key]:
                raise ValueError(f"Worker {index} has inconsistent {key}")
        records = worker["patch_results"]
        ids = [r["id"] for r in records]
        if len(ids) != len(set(ids)) or len(ids) != worker["patches"]:
            raise ValueError(f"Duplicate IDs or patch count mismatch: {directory}")
        if set(ids) != set(expected[index::num_shards]):
            raise ValueError(f"Incomplete or duplicate patch coverage: {directory}")
        collection = json.loads((directory / "raw_detections.geojson").read_text())
        if collection["type"] != "FeatureCollection":
            raise ValueError(f"Invalid GeoJSON: {directory}")
        features = collection["features"]
        counts = Counter(feature["properties"]["patch"] for feature in features)
        if (len(features) != worker["raw_instances"] or set(counts) - set(ids)
                or any(counts[r["id"]] != r["detections"] for r in records)):
            raise ValueError(f"Raw predictions disagree with patch counts: {directory}")
        objects.extend(features)
        patch_results.extend(records)
        workers.append(worker)
    ids = [r["id"] for r in patch_results]
    if Counter(ids) != Counter(expected):
        raise ValueError("Incomplete or duplicate global patch coverage")
    if objects:
        boxes = np.array([o["properties"]["pixel_box"] for o in objects], dtype=float)
        scores = np.array([o["properties"]["score"] for o in objects], dtype=float)
        if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
                or not np.isfinite(scores).all() or (boxes[:, 2:] <= boxes[:, :2]).any()):
            raise ValueError("Invalid prediction boxes or scores")
    kept = [objects[i] for i in suppress(objects)]
    by_id = {r["id"]: r for r in patch_results}
    summary = {key: workers[0][key] for key in common_fields}
    summary.update(
        num_shards=num_shards, patches=len(expected), raw_instances=len(objects),
        instances_after_nms=len(kept),
        nms="greedy mask intersection over smaller area > 0.5; masks clear of tile edges first, then score",
        inference_and_output_worker_seconds=sum(w["inference_and_output_seconds"] for w in workers),
        longest_worker_inference_and_output_seconds=max(w["inference_and_output_seconds"] for w in workers),
        load_worker_seconds=sum(w["load_seconds"] for w in workers),
        workers=[{k: v for k, v in w.items() if k != "patch_results"} for w in workers],
        patch_results=[by_id[patch_id] for patch_id in expected],
        validation="All scored patches processed exactly once; model predictions, not verified counts",
    )
    with Image.open(screen / "overview.jpg") as image:
        overview = image.convert("RGB")
    height, width = meta["shape"]
    sx, sy = overview.width / width, overview.height / height
    draw = ImageDraw.Draw(overview)
    for feature in kept:
        x1, y1, x2, y2 = feature["properties"]["pixel_box"]
        draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline="#00ff66", width=1)
    out.mkdir(parents=True, exist_ok=True)
    for name, features in (("raw_detections", objects), ("detections", kept)):
        (out / f"{name}.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": features}))
    overview.save(out / "detections_overview.jpg", quality=95)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--screen", type=Path, default=Path("results/screen"))
    parser.add_argument("--out", type=Path, default=Path("results/gc1_merged"))
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    summary = merge_shards(args.shards, args.screen, args.out, args.num_shards)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("workers", "patch_results")}, indent=2))


if __name__ == "__main__":
    main()
