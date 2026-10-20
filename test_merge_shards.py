import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from merge_shards import merge_shards


def detection(patch, score, box, edge=False):
    x1, y1, x2, y2 = box
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]]},
            "properties": {"patch": patch, "score": score, "pixel_box": box, "touches_patch_edge": edge}}


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.screen, self.shards, self.out = [self.root / name for name in ("screen", "shards", "out")]
        self.screen.mkdir()
        (self.screen / "summary.json").write_text(json.dumps(
            {"source": "input.tif", "shape": [100, 100]}))
        (self.screen / "patches.csv").write_text("id,status\np0,scored\np1,scored\np2,nodata\n")
        Image.new("RGB", (100, 100)).save(self.screen / "overview.jpg")
        for i in range(2):
            directory = self.shards / f"shard-{i}"
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(json.dumps(dict(
                model="facebook/sam3", prompt="circular storage tank", threshold=0.3,
                selection="all_valid", dtype="bfloat16", image="input.tif",
                rgb_bands=[1, 2, 3], shard_index=i, num_shards=2, batch_size=4,
                patches=1, raw_instances=1, load_seconds=1,
                inference_and_output_seconds=2 + i,
                patch_results=[{"id": f"p{i}", "detections": 1}],
            )))
            self.write_raw(i, detection(f"p{i}", 0.9 - i * 0.1, [10 + i, 10, 30 + i, 30]))

    def write_raw(self, index, feature):
        (self.shards / f"shard-{index}" / "raw_detections.geojson").write_text(json.dumps(
            {"type": "FeatureCollection", "features": [feature]}))

    def merge(self):
        return merge_shards(self.shards, self.screen, self.out, 2)

    def test_cross_shard_duplicates_are_suppressed_from_raw(self):
        result = self.merge()
        self.assertEqual(result["patches"], 2)
        self.assertEqual(result["raw_instances"], 2)
        self.assertEqual(result["instances_after_nms"], 1)
        self.assertEqual(result["inference_and_output_worker_seconds"], 5)
        kept = json.loads((self.out / "detections.geojson").read_text())["features"]
        self.assertEqual(kept[0]["properties"]["patch"], "p0")
        self.assertTrue((self.out / "detections_overview.jpg").is_file())

    def test_whole_mask_suppresses_higher_scoring_clipped_fragment(self):
        # Box IoU is 0.3, so box NMS kept both; the fragment lies wholly inside the whole mask.
        self.write_raw(1, detection("p1", 0.95, [24, 10, 30, 30], edge=True))
        result = self.merge()
        self.assertEqual(result["instances_after_nms"], 1)
        kept = json.loads((self.out / "detections.geojson").read_text())["features"]
        self.assertEqual(kept[0]["properties"]["patch"], "p0")

    def test_separate_masks_are_kept(self):
        self.write_raw(1, detection("p1", 0.95, [30, 10, 50, 30], edge=True))
        self.assertEqual(self.merge()["instances_after_nms"], 2)

    def test_missing_worker_is_rejected(self):
        (self.shards / "shard-1" / "summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.merge()
        self.assertFalse(self.out.exists())

    def test_missing_or_duplicate_patches_are_rejected(self):
        path = self.shards / "shard-1" / "summary.json"
        worker = json.loads(path.read_text())
        for records in ([], [{"id": "p0", "detections": 1}]):
            with self.subTest(records=records):
                worker["patch_results"] = records
                worker["patches"] = len(records)
                path.write_text(json.dumps(worker))
                with self.assertRaisesRegex(ValueError, "coverage"):
                    self.merge()
                self.assertFalse(self.out.exists())

    def test_inconsistent_settings_are_rejected(self):
        path = self.shards / "shard-1" / "summary.json"
        worker = json.loads(path.read_text())
        worker["dtype"] = "float32"
        path.write_text(json.dumps(worker))
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.merge()


if __name__ == "__main__":
    unittest.main()
