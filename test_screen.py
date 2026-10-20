import unittest

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window

from screen import footprint, read_rgb, starts


class GeometryTests(unittest.TestCase):
    def test_every_pixel_and_edges_are_covered(self):
        for length in (1, 511, 512, 513, 11048, 11774):
            positions = starts(length, 512, 128)
            covered = np.zeros(length, dtype=bool)
            for x in positions:
                covered[x:x+512] = True
            self.assertTrue(covered.all())
            self.assertEqual(len(positions), len(set(positions)))
            self.assertEqual(positions[-1], max(0, length-512))

    def test_overlap_contains_objects_up_to_overlap_width(self):
        positions = starts(2000, 512, 128)
        for left in range(2000-128+1):
            self.assertTrue(any(x <= left and left+128 <= x+512 for x in positions))

    def test_bad_geometry(self):
        for args in [(0, 512, 128), (20, 0, 0), (100, 50, 50), (100, 50, -1)]:
            with self.assertRaises(ValueError):
                starts(*args)

    def test_rgb_and_geographic_coordinates(self):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=20, height=20, count=4, dtype="uint8",
                          crs="EPSG:4326", transform=from_origin(38, 24, .001, .001)) as src:
                data = np.full((4, 20, 20), 255, dtype=np.uint8)
                data[0], data[1], data[2] = 10, 20, 30
                src.write(data)
                image, _ = read_rgb(src, Window(2, 3, 5, 6))
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (5, 6))
                self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
                ring = footprint(src, 2, 3, 5, 6)["coordinates"][0]
                np.testing.assert_allclose(ring[0], [38.002, 23.997])
                np.testing.assert_allclose(ring[2], [38.007, 23.991])
                self.assertEqual(ring[0], ring[-1])

    def test_audit_includes_rejected_patches_without_spatial_duplicates(self):
        from detect import select_audit
        rows = [
            dict(id=str(i), x=i*512, y=0, width=512, height=512,
                 margin=m, run_detector=m >= -0.02)
            for i, m in enumerate([0.08, 0.06, 0.04, 0.02,
                                   -0.001, -0.003, -0.005, -0.01,
                                   -0.021, -0.03, -0.04, -0.05])
        ]
        selected = select_audit(rows)
        self.assertEqual(len(selected), 12)
        self.assertEqual(sum(not r["run_detector"] for r in selected), 4)
        self.assertEqual(len({r["id"] for r in selected}), 12)


if __name__ == "__main__":
    unittest.main()
