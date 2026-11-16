"""Export finished results as static files for the interactive map in app/ (no GPU).

Writes a JPEG of the scene (bands 1-3) with its EPSG:3857 bounds, WGS84 GeoJSON layers for the
512 px gate decisions, SAM 3 detections and reference labels, and a summary.json of the
recorded metrics, which the app reads its counts from.
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
from shapely.geometry import mapping

from evaluate import project
from patch_decisions import LABELS
from validate_tiles import MODELS

TILE_FIELDS = ["patch_id", "in_extent", "truth", *MODELS]
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


def check_gate(tile_features, detections, gate):
    """Every detection must come from a tile that `gate` routed to SAM 3; else the flag names the wrong run."""
    decisions = {f["properties"]["patch_id"]: f["properties"][gate] for f in tile_features}
    wrong = sorted({d["properties"]["patch"] for d in detections
                    if decisions.get(d["properties"]["patch"], LABELS[2]) == LABELS[2]})
    if wrong:
        raise ValueError(f"--gate {gate} did not route tiles {wrong[:5]} that have detections; wrong --gate?")


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
    p.add_argument("--validation", type=Path, required=True, help="evaluate.py folder of the featured SAM 3 run")
    p.add_argument("--gate", required=True, choices=MODELS, help="model whose tile decisions fed that run")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a fresh output directory")
    for path in [a.image, a.tiles_gpkg, a.validation]:
        if not path.exists():
            p.error(f"Missing input: {path}")

    extent_features = read_features(a.validation / "extent.geojson")
    validation = json.loads((a.validation / "summary.json").read_text())

    with rasterio.open(a.image) as src:
        if str(src.crs) != "EPSG:3857":
            raise ValueError(f"Expected an EPSG:3857 image, got {src.crs}")
        b = src.bounds
        scene = dict(xmin=b.left, ymin=b.bottom, xmax=b.right, ymax=b.top, wkid=3857,
                     width=src.width, height=src.height)
        # Read every input before writing anything, so a bad input leaves --out empty.
        tile_layer = tiles(a.tiles_gpkg)
        detections = read_features(a.validation / "predictions.geojson")
        check_gate(tile_layer["features"], detections, a.gate)
        layers = {
            "tiles_512.geojson": tile_layer,
            "detections.geojson": collection(detections),
            "truth.geojson": collection(read_features(a.validation / "ground_truth.geojson")),
            "extent.geojson": collection(extent_features),
        }
        summary = dict(
            scene=scene,
            gate=a.gate,
            tiles=run_summary(a.tiles_gpkg),
            # Feature `status` (TP/FP, matched/FN) comes from the IoU >= 0.5 matching.
            detections=dict(ground_truth_evaluated=validation["ground_truth_evaluated"],
                            iou_0_50=validation["evaluations"]["iou_0.50"]),
        )
        a.out.mkdir(parents=True, exist_ok=True)
        scene["downsample"] = scale = scene_jpeg(src, a.out / "scene.jpg")
    for name, data in layers.items():
        (a.out / name).write_text(json.dumps(data))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Wrote {len(layers) + 2} files to {a.out} (scene downsample {scale}x)")


if __name__ == "__main__":
    main()
