from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pyogrio.raw import write
import numpy as np
import shapely
from shapely.geometry import box, shape

from export_web import check_gate, run_summary, tiles


def tile(patch_id, decision):
    return {"properties": {"patch_id": patch_id, "qwen32b": decision}}


class GateTests(unittest.TestCase):
    def test_detections_on_routed_tiles_pass(self):
        check_gate([tile("p0", "Contains a tank"), tile("p1", "Partially contains a tank")],
                   [{"properties": {"patch": "p0"}}, {"properties": {"patch": "p1"}}], "qwen32b")

    def test_detection_on_a_no_tank_or_unknown_tile_names_the_wrong_gate(self):
        for patch in ("p0", "missing"):
            with self.assertRaises(ValueError):
                check_gate([tile("p0", "Does not contain a tank")], [{"properties": {"patch": patch}}], "qwen32b")


class GeoPackageTests(unittest.TestCase):
    def test_tiles_are_reprojected_and_summary_is_filtered_by_size(self):
        with TemporaryDirectory() as d:
            gpkg = str(Path(d) / "t.gpkg")
            fields = ["patch_id", "in_extent", "truth", "qwen4b", "qwen32b", "nemotron12b", "nemotron30b"]
            values = [np.array(["p0"], dtype=object), np.array([1]), *[np.array(["Contains a tank"], dtype=object)] * 5]
            write(gpkg, np.array([shapely.to_wkb(box(0, 0, 111319.49, 111325.14))], dtype=object), values, fields,
                  layer="tiles_512", geometry_type="Polygon", crs="EPSG:3857", driver="GPKG")
            write(gpkg, None, [np.array([256, 512]), np.array(["a", "b"], dtype=object)], ["tile_size", "model"],
                  layer="run_summary", driver="GPKG", append=True)
            out = tiles(gpkg)["features"][0]
            self.assertEqual(out["properties"]["patch_id"], "p0")
            self.assertEqual(out["properties"]["in_extent"], 1)
            self.assertAlmostEqual(shape(out["geometry"]).bounds[2], 1.0, places=4)
            self.assertEqual(list(run_summary(gpkg)), ["b"])


if __name__ == "__main__":
    unittest.main()
