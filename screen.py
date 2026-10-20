"""Training-free patch screening; scores are similarities, not probabilities."""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window


POSITIVE = [
    "An aerial photograph of a storage tank.",
    "A satellite image of circular oil storage tanks.",
    "An overhead image of a tank farm.",
    "An aerial photograph of a dark circular storage tank.",
]
NEGATIVE = [
    "An aerial photograph of industrial buildings and factory roofs.",
    "An aerial photograph of refinery pipes and processing equipment.",
    "An aerial photograph of bare ground and desert.",
    "An aerial photograph of roads and parking lots.",
    "An aerial photograph of ocean water and coastline.",
    "An aerial photograph of rectangular water treatment ponds.",
    "An aerial photograph of ships and docks.",
]


def starts(length, size, overlap):
    if length <= 0 or size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require positive dimensions and 0 <= overlap < size")
    last = max(0, length - size)
    return sorted(set([*range(0, last + 1, size - overlap), last]))


def read_rgb(src, window):
    """The fourth band is never passed to an image model."""
    if src.count < 3 or any(t != "uint8" for t in src.dtypes[:3]):
        raise ValueError("This experiment requires three uint8 RGB bands")
    rgb = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
    valid = src.dataset_mask(window=window) > 0
    rgb[~valid] = 127
    return Image.fromarray(rgb), valid


def footprint(src, x, y, w, h):
    xy = [src.transform * p for p in
          [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]]
    lon, lat = transform_coords(src.crs, "EPSG:4326", *zip(*xy))
    return {"type": "Polygon", "coordinates": [list(map(list, zip(lon, lat)))]}


def pixel_metres(src):
    x, y = src.width / 2, src.height / 2
    xy = [src.transform * p for p in [(x, y), (x+1, y), (x, y+1)]]
    lon, lat = transform_coords(src.crs, "EPSG:4326", *zip(*xy))
    # Local spherical approximation is adequate for choosing patch size.
    scale = math.pi * 6371008.8 / 180
    return [scale * math.hypot((lon[i]-lon[0]) * math.cos(math.radians(lat[0])),
                              lat[i]-lat[0]) for i in (1, 2)]


def write_visuals(src, rows, out):
    width = 1500
    height = round(src.height * width / src.width)
    rgb = src.read([1, 2, 3], out_shape=(3, height, width),
                   resampling=Resampling.bilinear).transpose(1, 2, 0)
    image = Image.fromarray(rgb)
    image.save(out / "overview.jpg")
    draw = ImageDraw.Draw(image)
    sx, sy = width / src.width, height / src.height
    for r in rows:
        if not r["run_detector"]:
            continue
        color = "#28e070" if r["answer"] == "yes" else "#ffbd42"
        draw.rectangle((r["x"]*sx, r["y"]*sy,
                        (r["x"]+r["width"])*sx, (r["y"]+r["height"])*sy),
                       outline=color, width=2)
    draw.rectangle((0, 0, width, 28), fill="black")
    draw.text((8, 8), "Green: YES | Amber: uncertain, also run detector | Not verified detections",
              fill="white")
    image.save(out / "screening.jpg")
    valid = [r for r in rows if r["margin"] is not None]
    ordered = sorted(valid, key=lambda r: r["margin"], reverse=True)
    groups = [("Highest scores", ordered[:8]),
              ("Near decision boundary", sorted(valid, key=lambda r: abs(r["margin"]))[:8]),
              ("Lowest scores", ordered[-8:])]
    sheet = Image.new("RGB", (8*192, 3*236), "#171717")
    draw = ImageDraw.Draw(sheet)
    for row, (title, records) in enumerate(groups):
        for col, r in enumerate(records):
            chip, _ = read_rgb(src, Window(r["x"], r["y"], r["width"], r["height"]))
            sheet.paste(chip.resize((192, 192)), (col*192, row*236+24))
            draw.text((col*192+3, row*236+218),
                      f'{r["id"]}  margin {r["margin"]:+.3f}', fill="white")
        draw.text((3, row*236+5), title, fill="white")
    sheet.save(out / "contact_sheet.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/RemoteCLIP-ViT-B-32.pt"))
    parser.add_argument("--out", type=Path, default=Path("results/screen"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-margin", type=float, default=-0.02)
    args = parser.parse_args()
    starts(1, args.size, args.overlap)
    if args.batch_size < 1 or not math.isfinite(args.route_margin) or args.route_margin > 0:
        parser.error("batch-size must be positive and route-margin finite and <= 0")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "patches.csv").exists():
        parser.error("Output already exists; choose a new --out directory")

    import open_clip
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(4)
    start = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.inference_mode():
        text = model.encode_text(tokenizer(POSITIVE + NEGATIVE).to(device))
        text /= text.norm(dim=-1, keepdim=True)
    load_seconds = time.perf_counter() - start
    rows, batch, pending = [], [], []

    def flush():
        if not batch:
            return
        with torch.inference_mode():
            features = model.encode_image(torch.stack(batch).to(device))
            features /= features.norm(dim=-1, keepdim=True)
            sims = (features @ text.T).cpu().numpy()
        for r, sim in zip(pending, sims):
            pos, neg = float(sim[:len(POSITIVE)].max()), float(sim[len(POSITIVE):].max())
            margin = pos - neg
            r.update(positive_similarity=pos, negative_similarity=neg, margin=margin,
                     answer="yes" if margin >= 0 else "no",
                     run_detector=margin >= args.route_margin)
        batch.clear()
        pending.clear()
        print(f"Scored through patch {rows[-1]['id']}", flush=True)

    start = time.perf_counter()
    with rasterio.open(args.image) as src:
        if src.crs is None:
            raise ValueError("A CRS is required for geographic output")
        for y in starts(src.height, args.size, args.overlap):
            for x in starts(src.width, args.size, args.overlap):
                w, h = min(args.size, src.width), min(args.size, src.height)
                image, mask = read_rgb(src, Window(x, y, w, h))
                r = dict(id=f"p{len(rows):04}", x=x, y=y, width=w, height=h,
                         valid_fraction=float(mask.mean()), positive_similarity=None,
                         negative_similarity=None, margin=None, answer="no",
                         run_detector=False, status="scored" if mask.any() else "nodata")
                rows.append(r)
                if mask.any():
                    batch.append(preprocess(image))
                    pending.append(r)
                    if len(batch) == args.batch_size:
                        flush()
        flush()
        screen_seconds = time.perf_counter() - start
        with (args.out / "patches.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        features = [{"type": "Feature", "properties": r,
                     "geometry": footprint(src, r["x"], r["y"], r["width"], r["height"])}
                    for r in rows]
        (args.out / "patches.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": features}))
        summary = dict(source=str(args.image.resolve()), shape=[src.height, src.width],
                       rgb_bands=[1, 2, 3], mask="dataset validity mask; not model input",
                       pixel_size_metres_approx=pixel_metres(src),
                       patch_size=args.size, overlap_pixels=args.overlap,
                       total_patches=len(rows), scored=sum(r["status"] == "scored" for r in rows),
                       yes=sum(r["answer"] == "yes" for r in rows),
                       routed=sum(r["run_detector"] for r in rows),
                       route_margin=args.route_margin, device=device,
                       load_seconds=load_seconds, screening_seconds=screen_seconds,
                       model="RemoteCLIP ViT-B-32", checkpoint=str(args.checkpoint),
                       positive_prompts=POSITIVE, negative_prompts=NEGATIVE,
                       score_definition="max positive cosine - max negative cosine; not probability",
                       validation="No ground truth; thresholds are exploratory, not calibrated")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        write_visuals(src, rows, args.out)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
