import unittest
import hashlib
import json
from pathlib import Path
import csv
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin
from patch_decisions import collect, gate, grid_starts, prepare, LABELS


class PatchDecisionTests(unittest.TestCase):
    def test_fixed_half_overlap_and_complete_coverage(self):
        for length in (1, 511, 512, 513, 1024, 11048, 11774):
            starts = grid_starts(length)
            self.assertTrue(all(b-a == 256 for a, b in zip(starts, starts[1:])))
            covered = np.zeros(length, dtype=bool)
            for start in starts:
                covered[start:min(start+512, length)] = True
            self.assertTrue(covered.all())
            self.assertLess(starts[-1], length)

    def test_three_distinct_labels_and_invalid_grid(self):
        self.assertEqual(len(set(LABELS)), 3)
        for length, size in ((0, 512), (100, 511), (100, 0)):
            with self.assertRaises(ValueError):
                grid_starts(length, size)

    def test_collection_rejects_missing_duplicate_or_invalid_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, shards = root / "manifest", root / "shards"
            manifest.mkdir()
            worker = shards / "shard-0"
            worker.mkdir(parents=True)
            path = manifest / "patches.geojson"
            path.write_text(json.dumps({"features": [{"properties": {"id": "p0"}},
                                                     {"properties": {"id": "p1"}}]}))
            (worker / "summary.json").write_text(json.dumps(dict(
                num_shards=1, shard_index=0, model="test", revision="test", prompt_sha256="test",
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            good = [dict(id="p0", decision=LABELS[0]), dict(id="p1", decision=LABELS[1])]
            output = worker / "decisions.jsonl"
            output.write_text("\n".join(map(json.dumps, good)))
            self.assertEqual(len(collect(manifest, shards)[1]), 2)
            for bad in [good[:1], good+[good[0]], [good[0], dict(id="p1", decision="unknown")]]:
                with self.subTest(bad=bad):
                    output.write_text("\n".join(map(json.dumps, bad)))
                    with self.assertRaises(ValueError):
                        collect(manifest, shards)


    def test_gate_routes_only_present_tiles_in_screen_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.tif"
            with rasterio.open(image, "w", driver="GTiff", width=600, height=600, count=3, dtype="uint8",
                               crs="EPSG:3857", transform=from_origin(0, 600, 1, 1)) as dst:
                dst.write(np.full((3, 600, 600), 90, dtype="uint8"))
            prepare(image, root / "manifest", size=512)
            ids = [f["properties"]["id"] for f in json.loads((root / "manifest" / "patches.geojson").read_text())["features"]]
            self.assertEqual(len(ids), 4)
            worker = root / "run" / "shard-0"
            worker.mkdir(parents=True)
            (worker / "summary.json").write_text(json.dumps(dict(
                num_shards=1, shard_index=0, model="test", revision="test", prompt_sha256="test",
                manifest_sha256=hashlib.sha256((root / "manifest" / "patches.geojson").read_bytes()).hexdigest())))
            weights = [[.7, .2, .1], [.1, .6, .3], [.1, .1, .8], [.2, .2, .6]]
            decisions = [dict(id=i, decision=LABELS[int(np.argmax(w))], candidate_weights=w) for i, w in zip(ids, weights)]
            (worker / "decisions.jsonl").write_text("\n".join(map(json.dumps, decisions)))
            gate(root / "manifest", root / "run", root / "gate")
            with (root / "gate" / "patches.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["id"] for r in rows], ids[:2])
            self.assertEqual([r["run_detector"] for r in rows], ["True", "True"])
            self.assertAlmostEqual(float(rows[0]["margin"]), .8)
            self.assertTrue((root / "gate" / "overview.jpg").exists())
            self.assertEqual(json.loads((root / "gate" / "summary.json").read_text())["routed"], 2)
            for d in decisions:
                del d["candidate_weights"]
            (worker / "decisions.jsonl").write_text("\n".join(map(json.dumps, decisions)))
            with self.assertRaises(ValueError):
                gate(root / "manifest", root / "run", root / "gate-no-weights")


if __name__ == "__main__":
    unittest.main()
