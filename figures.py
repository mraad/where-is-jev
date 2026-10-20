"""Render documentation figures from a finished gate, SAM 3 merge and evaluation (no GPU).

Example tiles are chosen by fixed rules, not by hand: the tile with the most true positives,
the tile with the most false positives, and the partial tile with the most raw masks.
"""

import argparse
import csv
import json
from pathlib import Path

import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import box, shape

from evaluate import project

GREEN, AMBER, MAGENTA, RED, CYAN, YELLOW, GREY = (
    "#3ddc84", "#ffb020", "#ff3dd9", "#ff3030", "#38c8ff", "#ffe140", "#b0b0b0")
FONT = ImageFont.load_default(size=16)


def load(path):
    return [(shape(f["geometry"]), f["properties"]) for f in json.loads(Path(path).read_text())["features"]]


def to_pixels(geometry, src):
    inverse = ~src.transform
    return affine_transform(project(geometry, "EPSG:4326", src.crs),
                            [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f])


def panel(src, view, width, layers):
    """Read the pixel box `view` at `width` pixels; draw (pixel geometry, color, filled) layers."""
    x0, y0, x1, y1 = (round(v) for v in view)
    scale = width / (x1 - x0)
    height = round((y1 - y0) * scale)
    rgb = src.read([1, 2, 3], window=Window(x0, y0, x1 - x0, y1 - y0), out_shape=(3, height, width),
                   resampling=Resampling.bilinear).transpose(1, 2, 0)
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for geometry, color, filled in layers:
        if not geometry.intersects(box(x0, y0, x1, y1)):
            continue
        geometry = affine_transform(geometry, [scale, 0, 0, scale, -x0 * scale, -y0 * scale])
        for polygon in getattr(geometry, "geoms", [geometry]):
            ring = list(polygon.exterior.coords)
            if filled:
                draw.polygon(ring, fill=color + "66")
            draw.line(ring, fill=color, width=2)
    return Image.alpha_composite(image, overlay).convert("RGB")


def strip(image, items):
    """Add a dark strip above the image: text, or a color swatch and text per item."""
    out = Image.new("RGB", (image.width, image.height + 30), "#111111")
    out.paste(image, (0, 30))
    draw = ImageDraw.Draw(out)
    x = 8
    for text, color in items:
        if color:
            draw.rectangle((x, 9, x + 12, 21), fill=color)
            x += 18
        draw.text((x, 6), text, fill="white", font=FONT)
        x += draw.textlength(text, font=FONT) + 18
    return out


def row(images):
    out = Image.new("RGB", (sum(i.width for i in images) + 8 * (len(images) - 1), max(i.height for i in images)), "white")
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + 8
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a fresh output directory")
    with (a.gate / "patches.csv").open() as f:
        tiles = list(csv.DictReader(f))
    predictions = {props["id"]: props["status"] for _, props in load(a.validation / "predictions.geojson")}
    with rasterio.open(a.image) as src:
        kept = [(to_pixels(g, src), predictions.get(i, "not scored")) for i, (g, _) in enumerate(load(a.merged / "detections.geojson"))]
        raw = [to_pixels(g, src) for g, _ in load(a.merged / "raw_detections.geojson")]
        missed = [(to_pixels(g, src), props) for g, props in load(a.validation / "ground_truth.geojson")
                  if props["status"] == "FN"]
        extent = to_pixels(load(a.validation / "extent.geojson")[0][0], src)
        color = {"TP": GREEN, "FP": MAGENTA, "not scored": GREY}
        scored = [(g, color[s], True) for g, s in kept] + [(g, RED, False) for g, _ in missed]
        a.out.mkdir(parents=True, exist_ok=True)

        view = extent.buffer(40).bounds
        gate = [(box(int(t["x"]), int(t["y"]), int(t["x"]) + int(t["width"]), int(t["y"]) + int(t["height"])),
                 AMBER if t["answer"] == "partial" else GREEN, False) for t in tiles]
        strip(panel(src, view, 1200, gate + [(extent, YELLOW, False)]),
              [("Contains a tank", GREEN), ("Partially contains a tank", AMBER), ("Extent", YELLOW),
               ("Not drawn: no tank", None)]).save(a.out / "gate_tiles.jpg", quality=85)
        strip(panel(src, view, 1200, scored + [(extent, YELLOW, False)]),
              [("TP, IoU >= 0.5", GREEN), ("FP", MAGENTA), ("Not scored", GREY), ("Missed label", RED),
               ("Extent", YELLOW)]).save(a.out / "detections.jpg", quality=85)

        def inside(geometry, tile):
            return tile.contains(geometry.centroid)

        candidates = []
        for t in tiles:
            tile = box(int(t["x"]), int(t["y"]), int(t["x"]) + int(t["width"]), int(t["y"]) + int(t["height"]))
            if extent.covers(tile):
                counts = {s: sum(inside(g, tile) and k == s for g, k in kept) for s in ("TP", "FP")}
                candidates.append((t, tile, counts, sum(inside(g, tile) for g in raw)))
        most_tp = max(candidates, key=lambda c: c[2]["TP"])
        most_fp = max((c for c in candidates if c is not most_tp), key=lambda c: c[2]["FP"])
        partial = max((c for c in candidates if c[0]["answer"] == "partial" and c not in (most_tp, most_fp)),
                      key=lambda c: c[3])
        for name, (t, tile, counts, raw_count) in (("example_most_tp", most_tp), ("example_most_fp", most_fp),
                                                  ("example_partial", partial)):
            row([strip(panel(src, tile.bounds, 480, [(g, CYAN, True) for g in raw]),
                       [(f'{t["id"]}: {t["decision"]} | raw masks {raw_count}', None)]),
                 strip(panel(src, tile.bounds, 480, scored),
                       [(f'After suppression: TP {counts["TP"]}, FP {counts["FP"]}', None)])
                 ]).save(a.out / f"{name}.jpg", quality=85)

        chips = []
        for geometry, props in missed:
            x0, y0, x1, y1 = geometry.bounds
            half = max(x1 - x0, y1 - y0) * 0.75 + 20
            cx, cy = geometry.centroid.x, geometry.centroid.y
            chips.append(strip(panel(src, (cx - half, cy - half, cx + half, cy + half), 300, scored),
                               [(f'OBJECTID {props["id"]}, best IoU {props["best_iou"]:.2f}', None)]))
        if chips:
            row(chips).save(a.out / "missed_tanks.jpg", quality=85)

    written = sorted(a.out.glob("*.jpg"))
    for path in written:
        with Image.open(path) as image:
            assert image.width <= 1200, path
        print(path, path.stat().st_size // 1024, "KiB")
    assert len(written) == 5 + bool(chips), written


if __name__ == "__main__":
    main()
