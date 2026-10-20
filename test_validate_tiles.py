import unittest

from shapely.geometry import box

from patch_decisions import LABELS
from validate_tiles import reference_labels


class ReferenceLabelTests(unittest.TestCase):
    def test_complete_beats_partial_and_edge_contact_is_not_partial(self):
        tiles = [box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10), box(30, 0, 40, 10)]
        tanks = [box(2, 2, 4, 4), box(9, 5, 11, 7), box(30, 0, 32, 2)]
        # Tile 0 holds a whole tank and part of another; tile 1 holds only a part;
        # tile 2 touches a tank along its edge only; tile 3 holds one exactly on its boundary.
        self.assertEqual(reference_labels(tiles, tanks), [LABELS[0], LABELS[1], LABELS[2], LABELS[0]])

    def test_no_tanks_means_none(self):
        self.assertEqual(reference_labels([box(0, 0, 1, 1)], [box(5, 5, 6, 6)]), [LABELS[2]])


if __name__ == "__main__":
    unittest.main()
