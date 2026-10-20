"""Evaluate saved candidate polygons against FileGDB labels inside the sample extent."""

import argparse
import csv
import hashlib
import json
import math
from statistics import median
from pathlib import Path

import pyogrio
from pyogrio.raw import read
from pyproj import Transformer
from shapely import STRtree, from_wkb, transform
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def project(geometry, source, target):
    return transform(geometry, Transformer.from_crs(source, target, always_xy=True).transform,
                     interleaved=False)


def check_polygon(geometry):
    if (geometry is None or geometry.is_empty or not geometry.is_valid
            or geometry.geom_type not in ("Polygon", "MultiPolygon")):
        raise ValueError("Expected nonempty, valid polygon geometry; repair source explicitly")
    return geometry


def read_layer(path, layer):
    meta, ids, geometries, _ = read(path, layer=layer, columns=[], return_fids=True)
    if not meta["crs"]:
        raise ValueError(f"Missing CRS: {layer}")
    return [(int(fid), check_polygon(g)) for fid, g in zip(ids, from_wkb(geometries))], meta["crs"]


def restrict(records, extent):
    """Select by original centroid inside extent, then clip for overlap measurement."""
    included, excluded = [], []
    for record in records:
        if extent is not None and not extent.covers(record["geometry"].centroid):
            excluded.append(record)
            continue
        clipped = record["geometry"] if extent is None else record["geometry"].intersection(extent)
        if clipped.area <= 0:
            excluded.append(record)
            continue
        included.append(dict(record, geometry=clipped,
                             boundary_clipped=not clipped.equals(record["geometry"]),
                             centroid_in_extent=extent is None or extent.covers(record["geometry"].centroid),
                             retained_fraction=clipped.area / record["geometry"].area))
    return included, excluded


def overlap_pairs(predictions, truth):
    tree = STRtree([r["geometry"] for r in truth])
    pairs = []
    for pi, prediction in enumerate(predictions):
        pg = prediction["geometry"]
        for gi in tree.query(pg, predicate="intersects"):
            gg = truth[gi]["geometry"]
            area = pg.intersection(gg).area
            if area <= 0:  # Touching polygon edges are not a detection.
                continue
            pairs.append(dict(prediction_index=pi, truth_index=int(gi),
                              intersection_m2=area, iou=area / (pg.area + gg.area - area),
                              truth_coverage=area / gg.area, prediction_coverage=area / pg.area))
    return pairs


def match(predictions, truth, pairs, minimum_iou, has_extent):
    """Confidence-ordered, one-to-one matching; each detection gets its best free label."""
    candidates = {}
    for pair in pairs:
        if pair["iou"] >= minimum_iou:
            candidates.setdefault(pair["prediction_index"], []).append(pair)
    used_truth, matches = set(), []
    for pi in sorted(range(len(predictions)), key=lambda i: (-predictions[i]["score"], predictions[i]["id"])):
        available = [p for p in candidates.get(pi, []) if p["truth_index"] not in used_truth]
        if available:
            best = max(available, key=lambda p: (p["iou"], -p["truth_index"]))
            matches.append(best)
            used_truth.add(best["truth_index"])
    tp = len(matches)
    precision = tp / len(predictions) if has_extent and predictions else None
    recall = tp / len(truth) if truth else None
    count = len(predictions) + len(truth)
    f1 = 2 * tp / count if has_extent and count else None
    return dict(tp=tp, fp=len(predictions)-tp if has_extent else None,
                fn=len(truth)-tp, precision=precision, recall=recall, f1=f1,
                matches=matches)


def write_geojson(path, records, crs):
    features = []
    for record in records:
        properties = {k: v for k, v in record.items() if k != "geometry"}
        features.append(dict(type="Feature", properties=properties,
                             geometry=mapping(project(record["geometry"], crs, "EPSG:4326"))))
    path.write_text(json.dumps(dict(type="FeatureCollection", features=features), allow_nan=False))


def render_validation(image_path, out, predictions, truth, extent, crs):
    import rasterio
    from PIL import Image, ImageDraw
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds

    with rasterio.open(image_path) as src:
        rows = [(project(p["geometry"], crs, src.crs), "#00cfff") for p in predictions]
        rows += [(project(g["geometry"], crs, src.crs),
                  "#ff4949" if g["status"] == "FN" else "#44ff66") for g in truth]
        if extent is not None:
            rows.append((project(extent, crs, src.crs), "#ffe140"))

        def panel(region, width):
            view = project(region, crs, src.crs)
            bounds = view.bounds
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
            height = max(1, round(width * window.height / window.width))
            rgb = src.read([1, 2, 3], window=window, out_shape=(3, height, width),
                           resampling=Resampling.bilinear).transpose(1, 2, 0)
            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)
            inverse = ~src.window_transform(window)
            for geometry, color in rows:
                if not geometry.intersects(view):
                    continue
                polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
                for polygon in polygons:
                    for ring in [polygon.exterior, *polygon.interiors]:
                        pixels = [inverse * point[:2] for point in ring.coords]
                        draw.line([(x * width / window.width, y * height / window.height)
                                   for x, y in pixels], fill=color, width=2)
            return image

        region = extent if extent is not None else unary_union([g["geometry"] for g in truth]).envelope
        overview = panel(region.buffer(40), 1600)
        draw = ImageDraw.Draw(overview)
        draw.rectangle((0, 0, overview.width, 26), fill="black")
        draw.text((8, 8), "Extent only | IoU >= 0.50 | Green: matched label | Red: missed label | Cyan: predictions | Yellow: extent",
                  fill="white")
        overview.save(out / "validation_overview.jpg", quality=95)
        missed = [g for g in truth if g["status"] == "FN"]
        if missed:
            sheet = Image.new("RGB", (min(5, len(missed)) * 300, math.ceil(len(missed) / 5) * 330), "#171717")
            draw = ImageDraw.Draw(sheet)
            for i, g in enumerate(missed):
                x1, y1, x2, y2 = g["geometry"].bounds
                half = max(x2-x1, y2-y1) / 2 + 45
                region = g["geometry"].centroid.buffer(half).envelope
                chip = panel(region, 300).resize((300, 300))
                x, y = (i % 5) * 300, (i // 5) * 330
                sheet.paste(chip, (x, y))
                draw.text((x+5, y+307), f'OBJECTID {g["id"]} | best IoU {g["best_iou"]:.3f}', fill="white")
            sheet.save(out / "missed_tanks.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb", type=Path, required=True)
    parser.add_argument("--detections", type=Path, default=Path("results/gc1_gpu/merged/detections.geojson"))
    parser.add_argument("--out", type=Path, default=Path("results/validation"))
    parser.add_argument("--image", type=Path, help="Optional source RGB TIFF for validation previews")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory is not empty; choose a new --out")
    layers = {name.casefold(): name for name, _ in pyogrio.list_layers(args.gdb)}
    gt_layer, extent_layer = layers["tanks"], layers["extent"]
    labels, gt_crs = read_layer(args.gdb, gt_layer)
    extents, extent_crs = read_layer(args.gdb, extent_layer)
    if not labels:
        raise ValueError("No ground truth tanks")
    center = project(unary_union([g for _, g in labels]).centroid, gt_crs, "EPSG:4326")
    zone = min(60, max(1, math.floor((center.x + 180) / 6) + 1))
    metric_crs = f"EPSG:{(32600 if center.y >= 0 else 32700) + zone}"
    extent = unary_union([project(g, extent_crs, metric_crs) for _, g in extents]) if extents else None
    truth = [dict(id=fid, geometry=project(g, gt_crs, metric_crs)) for fid, g in labels]
    document = json.loads(args.detections.read_text())
    if document["type"] != "FeatureCollection" or document.get("crs"):
        raise ValueError("Require RFC 7946 longitude/latitude GeoJSON")
    predictions = []
    for i, feature in enumerate(document["features"]):
        score = float(feature["properties"]["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid confidence score")
        predictions.append(dict(id=i, score=score, patch=feature["properties"].get("patch"),
                                geometry=project(check_polygon(shape(feature["geometry"])),
                                                 "EPSG:4326", metric_crs)))
    truth, outside_truth = restrict(truth, extent)
    predictions, outside_predictions = restrict(predictions, extent)
    if not truth:
        raise ValueError("Sample extent contains no labeled tanks")
    pairs = overlap_pairs(predictions, truth)
    evaluations = {name: match(predictions, truth, pairs, threshold, extent is not None)
                   for name, threshold in [("any_positive_area", 0), ("iou_0.25", .25), ("iou_0.50", .5)]}
    summary = dict(
        gdb=str(args.gdb.resolve()), detections=str(args.detections.resolve()),
        detections_sha256=hashlib.sha256(args.detections.read_bytes()).hexdigest(),
        ground_truth_layer=gt_layer, extent_layer=extent_layer,
        ground_truth_source_crs=gt_crs, extent_source_crs=extent_crs, area_crs=metric_crs,
        extent_features=len(extents), mode="sample_extent" if extent is not None else "recall_only_missing_extent",
        ground_truth_total=len(labels), ground_truth_evaluated=len(truth),
        ground_truth_outside=len(outside_truth), detections_total=len(document["features"]),
        detections_evaluated=len(predictions), detections_outside_ignored=len(outside_predictions),
        boundary_truth=sum(r["boundary_clipped"] for r in truth),
        boundary_predictions=sum(r["boundary_clipped"] for r in predictions),
        ground_truth_with_any_overlap=len({p["truth_index"] for p in pairs}),
        detections_with_any_overlap=len({p["prediction_index"] for p in pairs}),
        matching="Descending confidence; best IoU unmatched label; one-to-one; positive-area overlap only",
        extent_policy="Only original centroids inside or on extent; clip retained polygons for IoU",
        scope_note="Precision and recall use ONLY detections and ground truth inside the sample extent. "
                   "Excluded detections contribute neither true positives nor false positives; "
                   "excluded ground truth contributes no false negatives." if extent is not None else
                   "Missing extent: recall-only diagnostic over all supplied labels.",
        precision_assumption="All tanks inside the sample extent are labeled",
        thresholds_changed=False,
        evaluations={name: {k: v for k, v in result.items() if k != "matches"}
                     for name, result in evaluations.items()},
    )
    if extent is None:
        summary["limitation"] = "Extent is empty; precision, FP and F1 are withheld. All labels used for recall."
    primary = evaluations["iou_0.50"]["matches"]
    if primary:
        summary["matched_iou_median_at_0_50"] = median(p["iou"] for p in primary)
    matched_pred = {p["prediction_index"]: p for p in primary}
    matched_gt = {p["truth_index"]: p for p in primary}
    for pi, p in enumerate(predictions):
        m = matched_pred.get(pi)
        p.update(status="TP" if m else "FP" if extent is not None else "unmatched_unscored",
                 matched_truth_id=truth[m["truth_index"]]["id"] if m else None,
                 matched_iou=m["iou"] if m else None,
                 best_iou=max((v["iou"] for v in pairs if v["prediction_index"] == pi), default=0))
    for gi, g in enumerate(truth):
        m = matched_gt.get(gi)
        g.update(status="matched" if m else "FN",
                 matched_prediction_id=predictions[m["prediction_index"]]["id"] if m else None,
                 best_iou=max((v["iou"] for v in pairs if v["truth_index"] == gi), default=0),
                 any_overlap=any(v["truth_index"] == gi for v in pairs))
    summary["labels_with_no_overlap"] = [g["id"] for g in truth if not g["any_overlap"]]
    summary["labels_missed_at_iou_0_50"] = [g["id"] for g in truth if g["status"] == "FN"]
    summary["unmatched_at_iou_0_50"] = dict(
        with_label_overlap=sum(p["status"] != "TP" and p["best_iou"] > 0 for p in predictions),
        without_label_overlap=sum(p["status"] != "TP" and p["best_iou"] == 0 for p in predictions))
    args.out.mkdir(parents=True, exist_ok=True)
    write_geojson(args.out / "predictions.geojson", predictions, metric_crs)
    write_geojson(args.out / "ground_truth.geojson", truth, metric_crs)
    if extent is not None:
        write_geojson(args.out / "extent.geojson", [dict(id="sample_extent", geometry=extent)], metric_crs)
    with (args.out / "overlaps.csv").open("w") as f:
        columns = ["prediction_id", "truth_id", "iou", "truth_coverage", "prediction_coverage", "intersection_m2"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for p in pairs:
            writer.writerow(dict(prediction_id=predictions[p["prediction_index"]]["id"],
                                 truth_id=truth[p["truth_index"]]["id"],
                                 **{k: p[k] for k in columns[2:]}))
    with (args.out / "ground_truth.csv").open("w") as f:
        fields = [k for k in truth[0] if k != "geometry"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in truth)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if args.image:
        render_validation(args.image, args.out, predictions, truth, extent, metric_crs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
