import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from patch_decisions import collect, grid_starts, LABELS


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


if __name__ == "__main__":
    unittest.main()
