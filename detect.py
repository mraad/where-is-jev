"""SAM 3 on screened RGB windows, or a diagnostic sample including rejected windows."""

import argparse
import csv
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from torchvision.ops import nms
from transformers import Sam3Model, Sam3Processor

from screen import read_rgb


def select_audit(rows, per_group=4):
    groups = [
        sorted((r for r in rows if r["margin"] >= 0), key=lambda r: -r["margin"]),
        sorted((r for r in rows if r["run_detector"] and r["margin"] < 0),
               key=lambda r: abs(r["margin"])),
        sorted((r for r in rows if not r["run_detector"]), key=lambda r: -r["margin"]),
    ]
    chosen = []
    for group in groups:
        count = 0
        for r in group:
            if any(abs(r["x"]-s["x"]) < r["width"] and
                   abs(r["y"]-s["y"]) < r["height"] for s in chosen):
                continue
            chosen.append(r)
            count += 1
            if count == per_group:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("results/screen"))
    parser.add_argument("--out", type=Path, default=Path("results/sam3"))
    parser.add_argument("--audit", action="store_true", help="12 spatially separated diagnostic patches")
    parser.add_argument("--all-valid", action="store_true", help="Ungated baseline on every valid patch")
    parser.add_argument("--prompt", default="storage tank")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--image", type=Path, help="Override source path when running on another host")
    parser.add_argument("--model", default="facebook/sam3", help="Cached model ID or local checkpoint directory")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--limit", type=int, default=0, help="Optional patch limit for a throughput probe")
    args = parser.parse_args()
    if args.audit and args.all_valid:
        parser.error("Choose either audit or all-valid")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    if not 0 <= args.shard_index < args.num_shards or args.batch_size < 1 or args.limit < 0:
        parser.error("Require 0 <= shard-index < num-shards, batch-size > 0, limit >= 0")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Output already exists; choose a new --out directory")
    meta = json.loads((args.screen / "summary.json").read_text())
    if args.image:
        meta["source"] = str(args.image.resolve())
    with (args.screen / "patches.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "scored"]
    for r in rows:
        for key in ["x", "y", "width", "height"]:
            r[key] = int(r[key])
        r["margin"] = float(r["margin"])
        r["run_detector"] = r["run_detector"] == "True"
    rows = (select_audit(rows) if args.audit else
            rows if args.all_valid else [r for r in rows if r["run_detector"]])
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error("No patches selected")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.dtype == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bfloat16 requires a supporting CUDA GPU")
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = Sam3Model.from_pretrained(args.model, local_files_only=True).to(device).eval()
    processor = Sam3Processor.from_pretrained(args.model, local_files_only=True)
    load_seconds = time.perf_counter() - start
    objects, patch_results = [], []
    start = time.perf_counter()
    with rasterio.open(meta["source"]) as src:
        if [src.height, src.width] != meta["shape"]:
            raise ValueError("Source raster dimensions differ from the screening manifest")
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset + args.batch_size]
            chips = [read_rgb(src, Window(r["x"], r["y"], r["width"], r["height"]))
                     for r in batch_rows]
            tick = time.perf_counter()
            inputs = processor(images=[chip[0] for chip in chips],
                               text=[args.prompt] * len(chips), return_tensors="pt").to(device)
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bfloat16" else nullcontext():
                    outputs = model(**inputs)
                results = processor.post_process_instance_segmentation(
                    outputs, threshold=args.threshold, mask_threshold=0.5,
                    target_sizes=inputs["original_sizes"].tolist())
            # CPU copies complete GPU execution before recording batch timing.
            results = [{k: v.cpu() for k, v in result.items()} for result in results]
            inference_per_patch = (time.perf_counter()-tick) / len(chips)
            for r, (image, valid), result in zip(batch_rows, chips, results):
                tick = time.perf_counter()
                window = Window(r["x"], r["y"], r["width"], r["height"])
                process_patch(src, window, r, image, valid, result, args.out,
                              objects, patch_results, tick, inference_per_patch)
            del outputs, inputs, results
        inference_seconds = time.perf_counter()-start
        if objects:
            keep = nms(torch.tensor([o["properties"]["pixel_box"] for o in objects], dtype=torch.float32),
                       torch.tensor([o["properties"]["score"] for o in objects]), 0.5).tolist()
        else:
            keep = []
        # shortcut: box NMS can retain clipped duplicates; use mask IoU/IoS for production mosaics.
        kept = [objects[i] for i in keep]
        (args.out / "raw_detections.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": objects}))
        (args.out / "detections.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": kept}))
        summary = dict(model=args.model, prompt=args.prompt, threshold=args.threshold,
                       selection="audit" if args.audit else "all_valid" if args.all_valid else "routed",
                       image=meta["source"], rgb_bands=[1, 2, 3], device=device,
                       shard_index=args.shard_index, num_shards=args.num_shards,
                       batch_size=args.batch_size, dtype=args.dtype, limit=args.limit,
                       gpu_name=torch.cuda.get_device_name() if device == "cuda" else None,
                       peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                       load_seconds=load_seconds, inference_and_output_seconds=inference_seconds,
                       patches=len(rows), raw_instances=len(objects), instances_after_nms=len(kept),
                       patch_results=patch_results,
                       validation="Model predictions, not verified counts; no labeled evaluation")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        if args.audit:
            sheet = Image.new("RGB", (4*320, math.ceil(len(rows)/4)*350), "#171717")
            draw = ImageDraw.Draw(sheet)
            for i, r in enumerate(patch_results):
                x, y = (i % 4)*320, (i//4)*350
                with Image.open(args.out / f'{r["id"]}.jpg') as image:
                    sheet.paste(image.resize((320, 320)), (x, y))
                draw.text((x+4, y+325),
                          f'{r["id"]} margin={r["screen_margin"]:+.3f}, SAM={r["detections"]}',
                          fill="white")
            sheet.save(args.out / "audit.jpg")
        print(json.dumps({k: v for k, v in summary.items() if k != "patch_results"}, indent=2))


def process_patch(src, window, r, image, valid, result, out_dir,
                  objects, patch_results, tick, inference_per_patch):
    masks = result["masks"].cpu().numpy()
    scores = result["scores"].cpu().tolist()
    overlay = np.array(image).copy()
    draw_boxes = []
    for mask, score in zip(masks, scores):
        mask = mask.astype(bool) & valid
        yy, xx = np.nonzero(mask)
        if not len(xx):
            continue
        box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
        polygons = [geom["coordinates"] for geom, value in shapes(
            mask.astype("uint8"), mask=mask, transform=src.window_transform(window))
            if value == 1]
        geom = transform_geom(src.crs, "EPSG:4326",
                              {"type": "MultiPolygon", "coordinates": polygons})
        objects.append({
            "type": "Feature", "geometry": geom,
            "properties": dict(patch=r["id"], score=score, screen_margin=r["margin"],
                               screen_routed=r["run_detector"],
                               pixel_box=[box[0]+r["x"], box[1]+r["y"],
                                          box[2]+r["x"], box[3]+r["y"]],
                               touches_patch_edge=(box[0] == 0 or box[1] == 0 or
                                                   box[2] == r["width"] or box[3] == r["height"]))
        })
        overlay[mask] = (overlay[mask]*0.6 + np.array([0, 255, 80])*0.4).astype("uint8")
        draw_boxes.append((box, score))
    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    for box, score in draw_boxes:
        draw.rectangle(box, outline="#00ff66", width=2)
        draw.text((box[0], box[1]), f"{score:.2f}", fill="white", stroke_width=1,
                  stroke_fill="black")
    annotated.save(out_dir / f'{r["id"]}.jpg')
    image.save(out_dir / f'{r["id"]}_rgb.jpg')
    record = dict(id=r["id"], screen_margin=r["margin"],
                  screen_routed=r["run_detector"], detections=len(draw_boxes),
                  seconds=inference_per_patch + time.perf_counter()-tick)
    patch_results.append(record)
    print(json.dumps(record), flush=True)
    # Keep progress inspectable if a long local inference is interrupted.
    with (out_dir / "progress.jsonl").open("a") as f:
        f.write(json.dumps(record)+"\n")


if __name__ == "__main__":
    main()
