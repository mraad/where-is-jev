"""Prepare 50%-overlap RGB windows of a chosen size and export three-way visual decisions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import rasterio
from rasterio.windows import Window
from PIL import Image

from screen import footprint, read_rgb

LABELS = ("Contains a tank", "Partially contains a tank", "Does not contain a tank")
QUESTION = """Classify this RGB overhead satellite image patch for circular industrial storage tanks.
A: Contains at least one complete storage tank whose full circular footprint is visible inside the patch.
B: Contains part of one or more storage tanks cut off by a patch edge or gray missing-image area, but no complete tank.
C: Contains no visible storage tank, either complete or partial.
If both complete and partial tanks appear, choose A. There may be one tank or many.
Partial means spatially cut off, not uncertain. Roads, factory roofs, ships, ponds, and bare ground are not storage tanks.
Gray regions are missing image data. Judge visible evidence only.
Answer with exactly one letter: A, B, or C."""


def grid_starts(length, size=512):
    if length < 1 or size < 2 or size % 2:
        raise ValueError("Positive raster length and positive even patch size required")
    stride = size // 2
    return list(range(0, max(1, length - size + stride), stride))


def chip(src, row, size=512):
    image, _ = read_rgb(src, Window(row["x"], row["y"], row["width"], row["height"]))
    if image.size != (size, size):
        padded = Image.new("RGB", (size, size), (127, 127, 127))
        padded.paste(image, (0, 0))
        image = padded
    return image


def prepare(image, out, size=512):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory")
    tick = time.perf_counter()
    features, skipped, total = [], 0, 0
    with rasterio.open(image) as src:
        if src.crs is None:
            raise ValueError("Image requires a CRS")
        for y in grid_starts(src.height, size):
            for x in grid_starts(src.width, size):
                w, h = min(size, src.width-x), min(size, src.height-y)
                _, valid = read_rgb(src, Window(x, y, w, h))
                patch_id = f"p50_{total:04}"
                total += 1
                if not valid.any():
                    skipped += 1
                    continue
                row = dict(id=patch_id, x=x, y=y, width=w, height=h,
                           valid_fraction=float(valid.sum() / size**2))
                features.append(dict(type="Feature", properties=row,
                                     geometry=footprint(src, x, y, w, h)))
        meta = dict(source=str(image.resolve()), shape=[src.height, src.width],
                    patch_size=size, stride=size//2, overlap_fraction=.5, rgb_bands=[1, 2, 3],
                    total_windows=total, valid_windows=len(features), nodata_skipped=skipped,
                    edge_policy=f"Fixed {size//2}-pixel grid; outer windows padded gray to {size} pixels; footprints clipped to raster",
                    nodata_policy="Wholly invalid windows omitted, not classified as no tank",
                    image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                    prepare_seconds=time.perf_counter()-tick)
    out.mkdir(parents=True, exist_ok=True)
    (out / "patches.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=features)))
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def collect(manifest, shards):
    features = json.loads((manifest / "patches.geojson").read_text())["features"]
    expected = [f["properties"]["id"] for f in features]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate manifest IDs")
    records, summaries = [], []
    for directory in sorted(shards.glob("shard-*")):
        summary = json.loads((directory / "summary.json").read_text())
        records.extend(json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines())
        summaries.append(summary)
    if not summaries:
        raise ValueError("No completed workers")
    count = summaries[0]["num_shards"]
    if len(summaries) != count or {s["shard_index"] for s in summaries} != set(range(count)):
        raise ValueError("Missing or repeated workers")
    for s in summaries:
        for key in ("model", "revision", "prompt_sha256", "manifest_sha256", "num_shards"):
            if s[key] != summaries[0][key]:
                raise ValueError(f"Inconsistent worker {key}")
    digest = hashlib.sha256((manifest / "patches.geojson").read_bytes()).hexdigest()
    if summaries[0]["manifest_sha256"] != digest:
        raise ValueError("Workers used a different patch manifest")
    ids = [r["id"] for r in records]
    if Counter(ids) != Counter(expected):
        raise ValueError("Missing, duplicated, or unexpected patch decisions")
    if any(r["decision"] not in LABELS for r in records):
        raise ValueError("Unexpected decision")
    by_id = {r["id"]: r for r in records}
    return features, by_id, summaries


def export(manifest, shards, out):
    import numpy as np
    from pyogrio.raw import write, read
    from shapely.geometry import shape
    import sqlite3

    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory")
    features, decisions, summaries = collect(manifest, shards)
    rows = [dict(f["properties"], decision=decisions[f["properties"]["id"]]["decision"]) for f in features]
    out.mkdir(parents=True, exist_ok=True)
    name = out / "tank_decisions_50pct.gpkg"
    metadata = "Standalone visual classification; A=one or more complete tanks; B=partial tank(s) only; C=no visible tank. Complete takes priority. Model predictions, not verified labels."
    fields = ["patch_id", "decision", "valid_fraction"]
    write(name, np.array([shape(f["geometry"]).wkb for f in features], dtype=object),
          [np.array([r["id"] for r in rows], dtype=object),
           np.array([r["decision"] for r in rows], dtype=object),
           np.array([r["valid_fraction"] for r in rows])], fields,
          layer="patch_decisions", driver="GPKG", geometry_type="Polygon", crs="EPSG:4326",
          dataset_options={"VERSION": "1.3"}, layer_options={"SPATIAL_INDEX": "YES", "DESCRIPTION": metadata},
          layer_metadata={"MODEL": summaries[0]["model"], "REVISION": summaries[0]["revision"],
                          "DESCRIPTION": metadata})
    exported = [dict(type="Feature", properties=row, geometry=f["geometry"]) for f, row in zip(features, rows)]
    (out / "patch_decisions.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=exported)))
    summary = dict(model=summaries[0]["model"], revision=summaries[0]["revision"],
                   patch_count=len(rows), counts=dict(Counter(r["decision"] for r in rows)),
                   source_manifest=json.loads((manifest / "manifest.json").read_text()),
                   workers=summaries, labels=list(LABELS), decision_rule=metadata)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _, _, geometry, data = read(name, layer="patch_decisions")
    assert len(geometry) == len(rows)
    assert set(data[1]).issubset(LABELS)
    with sqlite3.connect(name) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({k: v for k, v in summary.items() if k not in ("workers", "source_manifest")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("image", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--size", type=int, default=512)
    exp = sub.add_parser("export")
    exp.add_argument("--manifest", type=Path, required=True)
    exp.add_argument("--shards", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.image, a.out, a.size)
    else:
        export(a.manifest, a.shards, a.out)


if __name__ == "__main__":
    main()
