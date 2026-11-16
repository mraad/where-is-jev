"""Export finished results as static files for the interactive map in app/ (no GPU).

Writes a JPEG of the scene (bands 1-3) with its EPSG:3857 bounds, WGS84 GeoJSON layers for the
512 px gate decisions, SAM 3 detections, reference labels and the Kev filter pilot, and a
summary.json of the recorded metrics that the app checks its own numbers against.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import rasterio
import shapely
from PIL import Image
from pyogrio.raw import read
from shapely.geometry import mapping, shape

from evaluate import project

TILE_FIELDS = ["patch_id", "in_extent", "truth", "qwen4b", "qwen32b", "nemotron12b", "nemotron30b"]
KEV_MODELS = ["vision_linear", "kev_zero_shot", "kev_tuned_head"]
MAX_JPEG_BYTES = 15_000_000


def collection(features):
    return {"type": "FeatureCollection", "features": features}


def feature(geometry, properties):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def read_features(path):
    return json.loads(Path(path).read_text())["features"]


def tiles(gpkg, layer="tiles_512"):
    meta, _, geometries, values = read(gpkg, layer=layer, columns=TILE_FIELDS)
    columns = dict(zip(meta["fields"], values))
    features = []
    for i, wkb in enumerate(geometries):
        props = {name: columns[name][i].item() if hasattr(columns[name][i], "item") else columns[name][i]
                 for name in TILE_FIELDS}
        features.append(feature(project(shapely.from_wkb(wkb), meta["crs"], "EPSG:4326"), props))
    return collection(features)


def run_summary(gpkg, size=512):
    meta, _, _, values = read(gpkg, layer="run_summary", read_geometry=False)
    rows = [dict(zip(meta["fields"], (v.item() if hasattr(v, "item") else v for v in row)))
            for row in zip(*values)]
    return {row["model"]: row for row in rows if row["tile_size"] == size}


def kev(predictions, scores):
    """Keep scored candidates only (buffer candidates were never scored), with their scores."""
    by_id = {r["id"]: r for r in scores}
    features = []
    for f in predictions:
        record = by_id.get(f["properties"]["id"])
        if record is None:
            continue
        props = {k: record[k] for k in ["id", "split", "original_tp", *KEV_MODELS]}
        props["original_tp"] = int(props["original_tp"])
        props["score"] = f["properties"]["score"]
        features.append(feature(shape(f["geometry"]), props))
    if len(features) != len(scores):
        raise ValueError("Kev scores reference candidates missing from the predictions file")
    return collection(features)


def band(cuts, src, extent):
    """Test band: pixel columns cuts[2]..cuts[3] over the full image height, clipped to the extent."""
    left, top = src.transform @ (cuts[2], 0)
    right, bottom = src.transform @ (cuts[3], src.height)
    box = project(shapely.box(left, bottom, right, top), src.crs, "EPSG:4326")
    return collection([feature(box.intersection(extent), {"split": "test"})])


def scene_jpeg(src, out):
    """Halve the resolution until the JPEG fits MAX_JPEG_BYTES; encode in memory, write once."""
    image, scale = Image.fromarray(np.moveaxis(src.read([1, 2, 3]), 0, -1)), 1
    while True:
        data = io.BytesIO()
        image.resize((src.width // scale, src.height // scale), Image.LANCZOS).save(data, "JPEG", quality=80)
        if data.tell() <= MAX_JPEG_BYTES:
            out.write_bytes(data.getvalue())
            return scale
        scale *= 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--tiles-gpkg", type=Path, required=True)
    p.add_argument("--validation", type=Path, default=Path("results/sam3_round_validation"))
    p.add_argument("--kev-inputs", type=Path, default=Path("results/kev_filter_inputs/candidate/evaluation-retry"))
    p.add_argument("--kev-pilot", type=Path, default=Path("results/kev_filter_pilot/pilot-v1-recount"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a fresh output directory")
    for path in [a.image, a.tiles_gpkg, a.validation, a.kev_inputs, a.kev_pilot]:
        if not path.exists():
            p.error(f"Missing input: {path}")

    extent_features = read_features(a.validation / "extent.geojson")
    extent = shape(extent_features[0]["geometry"])
    report = json.loads((a.kev_pilot / "report.json").read_text())
    cuts = json.loads((a.kev_pilot / "records.json").read_text())["cuts"]
    validation = json.loads((a.validation / "summary.json").read_text())

    with rasterio.open(a.image) as src:
        if str(src.crs) != "EPSG:3857":
            raise ValueError(f"Expected an EPSG:3857 image, got {src.crs}")
        b = src.bounds
        scene = dict(xmin=b.left, ymin=b.bottom, xmax=b.right, ymax=b.top, wkid=3857,
                     width=src.width, height=src.height)
        # Read every input before writing anything, so a bad input leaves --out empty.
        layers = {
            "tiles_512.geojson": tiles(a.tiles_gpkg),
            "detections.geojson": collection(read_features(a.validation / "predictions.geojson")),
            "truth.geojson": collection(read_features(a.validation / "ground_truth.geojson")),
            "extent.geojson": collection(extent_features),
            "kev.geojson": kev(read_features(a.kev_inputs / "predictions.geojson"),
                               json.loads((a.kev_pilot / "scores.json").read_text())),
            "kev_band.geojson": band(cuts, src, extent),
        }
        summary = dict(
            scene=scene,
            tiles=run_summary(a.tiles_gpkg),
            # Feature `status` (TP/FP, matched/FN) comes from the IoU >= 0.5 matching.
            detections=dict(ground_truth_evaluated=validation["ground_truth_evaluated"],
                            iou_0_50=validation["evaluations"]["iou_0.50"]),
            kev=dict(thresholds={m: report["models"][m]["threshold"] for m in KEV_MODELS},
                     test_reference_tanks=report["test_reference_tanks"],
                     detection_evaluation=report["detection_evaluation"], scope=report["scope"]),
        )
        a.out.mkdir(parents=True, exist_ok=True)
        scene["downsample"] = scale = scene_jpeg(src, a.out / "scene.jpg")
    for name, data in layers.items():
        (a.out / name).write_text(json.dumps(data))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {len(layers) + 2} files to {a.out} (scene downsample {scale}x)")


if __name__ == "__main__":
    main()
