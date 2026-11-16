from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from affine import Affine
from pyogrio.raw import write
import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import box, mapping, shape

from export_web import band, kev, run_summary, tiles


def prediction(ident, geometry):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": {"id": ident, "score": 0.5}}


def score(ident):
    return dict(id=ident, split="test", original_tp=True, vision_linear=0.9, kev_zero_shot=0.2,
                kev_tuned_head=0.3)


class KevTests(unittest.TestCase):
    def test_keeps_only_scored_candidates_with_their_scores(self):
        out = kev([prediction(0, box(0, 0, 1, 1)), prediction(1, box(2, 2, 3, 3))], [score(1)])
        self.assertEqual(len(out["features"]), 1)
        props = out["features"][0]["properties"]
        self.assertEqual((props["id"], props["original_tp"], props["vision_linear"], props["score"]),
                         (1, 1, 0.9, 0.5))

    def test_scores_without_a_candidate_are_rejected(self):
        with self.assertRaises(ValueError):
            kev([prediction(0, box(0, 0, 1, 1))], [score(0), score(7)])


class BandTests(unittest.TestCase):
    def test_band_spans_the_cut_columns_and_is_clipped_to_the_extent(self):
        # 1 m pixels, origin at (0, 100) in EPSG:3857, 100 rows; cuts select columns 50-75.
        src = SimpleNamespace(transform=Affine(1, 0, 0, 0, -1, 100), height=100, crs="EPSG:3857")
        extent = scale(box(0, 20, 100, 100), 1e-5, 1e-5, origin=(0, 0))
        geometry = shape(band([0, 25, 50, 75], src, extent)["features"][0]["geometry"])
        xmin, ymin, xmax, _ = geometry.bounds
        self.assertAlmostEqual(xmin, 50 * 360 / 40075016.686, places=7)
        self.assertAlmostEqual(xmax, 75 * 360 / 40075016.686, places=7)
        self.assertAlmostEqual(ymin, 20e-5)


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
