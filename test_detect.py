import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window

from detect import process_patch


class PatchOutputTests(unittest.TestCase):
    def test_mask_clipping_global_offsets_and_progress(self):
        masks = torch.zeros((2, 8, 8), dtype=torch.bool)
        masks[0, 1:4, 2:5] = True
        masks[1, :, 0] = True
        valid = np.ones((8, 8), dtype=bool)
        valid[:, :3] = False
        row = dict(id="test", x=10, y=20, width=8, height=8,
                   margin=.1, run_detector=True)
        objects, records = [], []
        with tempfile.TemporaryDirectory() as tmp, MemoryFile() as mem:
            with mem.open(driver="GTiff", width=64, height=64, count=3, dtype="uint8",
                          crs="EPSG:4326", transform=from_origin(38, 24, .001, .001)) as src:
                with contextlib.redirect_stdout(io.StringIO()):
                    process_patch(src, Window(10, 20, 8, 8), row,
                                  Image.new("RGB", (8, 8)), valid,
                                  {"masks": masks, "scores": torch.tensor([.9, .8])},
                                  Path(tmp), objects, records, time.perf_counter(), .25)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0]["properties"]["pixel_box"], [13, 21, 15, 24])
            self.assertEqual(records[0]["detections"], 1)
            self.assertGreaterEqual(records[0]["seconds"], .25)
            self.assertEqual(json.loads((Path(tmp) / "progress.jsonl").read_text()), records[0])
            ring = objects[0]["geometry"]["coordinates"][0][0]
            np.testing.assert_allclose([min(p[0] for p in ring), max(p[1] for p in ring)],
                                       [38.013, 23.979])


if __name__ == "__main__":
    unittest.main()
