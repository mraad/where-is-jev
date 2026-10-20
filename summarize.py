"""Summarize the completed experiment; SAM agreement is not ground-truth recall."""

import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path("results")
    screen = json.loads((root / "screen/summary.json").read_text())
    sam = json.loads((root / "sam3_all/summary.json").read_text())
    features = json.loads((root / "sam3_all/detections.geojson").read_text())["features"]
    with (root / "screen/patches.csv").open() as f:
        rows = list(csv.DictReader(f))
    valid_ids = {r["id"] for r in rows if r["status"] == "scored"}
    patches = sam["patch_results"]
    assert len(rows) == screen["total_patches"]
    assert len(valid_ids) == screen["scored"]
    assert len(patches) == len(valid_ids) == len({r["id"] for r in patches})
    assert {r["id"] for r in patches} == valid_ids
    assert len(features) == sam["instances_after_nms"]
    for feature in features:
        p = feature["properties"]
        x1, y1, x2, y2 = p["pixel_box"]
        assert 0 <= x1 < x2 <= screen["shape"][1]
        assert 0 <= y1 < y2 <= screen["shape"][0]
        assert 0 <= p["score"] <= 1 and p["patch"] in valid_ids
    for name in ("jev_probe.json", "jev_simple.json"):
        for r in json.loads((root / name).read_text()):
            for answer in r["answers"].values():
                values = list(answer["probabilities"].values())
                assert all(math.isfinite(p) and 0 <= p <= 1 for p in values)
                assert math.isclose(sum(values), 1, abs_tol=1e-6)
    rejected = [r for r in patches if not r["screen_routed"] and r["detections"]]
    metrics = {
        "valid_patches": len(patches),
        "sam_positive_patches": sum(r["detections"] > 0 for r in patches),
        "sam_positive_patches_rejected_by_gate": len(rejected),
        "raw_sam_instances_in_rejected_patches": sum(r["detections"] for r in rejected),
        "sam_raw_instances": sam["raw_instances"],
        "sam_instances_after_nms": len(features),
        "sam_inference_and_output_seconds": sam["inference_and_output_seconds"],
        "interpretation": (
            "SAM predictions are not ground truth. Rejected-patch predictions may be false "
            "positives or duplicated in retained overlapping patches. These are not object-recall metrics."
        ),
    }
    (root / "comparison.json").write_text(json.dumps(metrics, indent=2))
    image = Image.open(root / "screen/overview.jpg").convert("RGB")
    draw = ImageDraw.Draw(image)
    sx, sy = image.width / screen["shape"][1], image.height / screen["shape"][0]
    for feature in features:
        x1, y1, x2, y2 = feature["properties"]["pixel_box"]
        draw.rectangle((x1*sx, y1*sy, x2*sx, y2*sy), outline="#00ff70", width=1)
    draw.rectangle((0, 0, image.width, 28), fill="black")
    draw.text((8, 8), f"SAM 3: {len(features)} candidate instances after box NMS | Unverified",
              fill="white")
    image.save(root / "detections_overview.jpg")
    audit = json.loads((root / "audit/summary.json").read_text())["patch_results"]
    groups = [
        ("Same 12 diagnostic patches; prompt: circular storage tank", audit, "sam3_examples.jpg"),
        ("Rejected by RemoteCLIP but SAM predicted objects; not verified misses",
         sorted(rejected, key=lambda r: -r["detections"])[:8], "rejected_examples.jpg"),
    ]
    for title, items, name in groups:
        sheet = Image.new("RGB", (4*320, math.ceil(len(items)/4)*350+28), "#171717")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 8), title, fill="white")
        by_id = {r["id"]: r for r in patches}
        for i, item in enumerate(items):
            r = by_id[item["id"]]
            x, y = (i % 4)*320, (i//4)*350+28
            with Image.open(root / "sam3_all" / f'{r["id"]}.jpg') as chip:
                sheet.paste(chip.resize((320, 320)), (x, y))
            draw.text((x+4, y+325),
                      f'{r["id"]} margin={r["screen_margin"]:+.3f}, SAM={r["detections"]}',
                      fill="white")
        sheet.save(root / name)
    print(json.dumps(metrics, indent=2))
    print("Artifact consistency checks passed.")


if __name__ == "__main__":
    main()
