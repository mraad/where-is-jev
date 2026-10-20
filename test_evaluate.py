import unittest

from shapely.geometry import Point, box

from evaluate import match, overlap_pairs, project, restrict


class EvaluationTests(unittest.TestCase):
    def test_projection_uses_longitude_latitude_order(self):
        point = project(Point(39, 0), "EPSG:4326", "EPSG:32637")
        self.assertAlmostEqual(point.x, 500000, places=5)
        self.assertAlmostEqual(point.y, 0, places=5)

    def test_extent_excludes_outside_and_clips_boundary(self):
        records = [dict(id=i, geometry=g) for i, g in enumerate(
            [box(1, 1, 2, 2), box(20, 20, 21, 21), box(9, 2, 12, 4),
             box(10, 1, 11, 2), box(8, 2, 11, 4)])]
        inside, outside = restrict(records, box(0, 0, 10, 10))
        self.assertEqual([r["id"] for r in inside], [0, 4])
        self.assertEqual([r["id"] for r in outside], [1, 2, 3])
        self.assertEqual(inside[1]["geometry"].area, 4)
        self.assertTrue(inside[1]["boundary_clipped"])
        self.assertTrue(all(r["centroid_in_extent"] for r in inside))

    def test_outside_detection_cannot_change_extent_metrics(self):
        extent = box(0, 0, 10, 10)
        truth, _ = restrict([dict(id=1, geometry=box(8, 2, 10, 4))], extent)
        predictions, excluded = restrict([
            dict(id=0, score=.9, geometry=box(8, 2, 10, 4)),
            dict(id=1, score=.99, geometry=box(9, 2, 12, 4)),
            dict(id=2, score=.99, geometry=box(20, 20, 22, 22)),
        ], extent)
        result = match(predictions, truth, overlap_pairs(predictions, truth), .5, True)
        self.assertEqual(len(excluded), 2)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 1)

    def test_duplicates_do_not_inflate_recall_and_edge_touch_is_not_overlap(self):
        truth = [dict(id=1, geometry=box(0, 0, 2, 2)), dict(id=2, geometry=box(4, 0, 6, 2))]
        predictions = [dict(id=i, score=s, geometry=g) for i, (s, g) in enumerate(
            [(.8, box(0, 0, 2, 2)), (.9, box(0, 0, 2, 2)), (.7, box(2, 0, 4, 2))])]
        pairs = overlap_pairs(predictions, truth)
        result = match(predictions, truth, pairs, .5, True)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 1))
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(len(pairs), 2)

    def test_overlap_and_iou_distinct_and_precision_withheld_without_extent(self):
        truth = [dict(id=1, geometry=box(0, 0, 10, 10))]
        predictions = [dict(id=0, score=.9, geometry=box(0, 0, 2, 2))]
        pairs = overlap_pairs(predictions, truth)
        loose = match(predictions, truth, pairs, 0, False)
        strict = match(predictions, truth, pairs, .5, False)
        self.assertEqual(loose["recall"], 1)
        self.assertEqual(strict["recall"], 0)
        self.assertIsNone(loose["fp"])
        self.assertIsNone(loose["precision"])
        self.assertIsNone(loose["f1"])


if __name__ == "__main__":
    unittest.main()
