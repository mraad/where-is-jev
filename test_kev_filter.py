import unittest
from experiments.kev_filter import exclude_buffered, partition, select_threshold

class FilterTests(unittest.TestCase):
    def test_split_excludes_crossing_crops(self):
        self.assertEqual(partition(10,40,[0,50,75,100]),"train")
        self.assertEqual(partition(45,55,[0,50,75,100]),"buffer")
        self.assertEqual(partition(80,95,[0,50,75,100]),"test")

    def test_threshold_preserves_calibration_positives(self):
        self.assertEqual(select_threshold([.1,.6,.9],[False,True,True]),.6)
        with self.assertRaises(ValueError):
            select_threshold([.1],[False])

    def test_label_found_only_by_a_buffer_candidate_is_not_scored(self):
        truth = [dict(id=81), dict(id=83), dict(id=90)]
        records = [dict(split="buffer", matched_truth_id=81), dict(split="test", matched_truth_id=90),
                   dict(split="buffer", matched_truth_id=None), dict(split="buffer", matched_truth_id=5)]
        kept, excluded = exclude_buffered(truth, records)
        self.assertEqual([r["id"] for r in kept], [83, 90])
        self.assertEqual(excluded, [81])

if __name__=="__main__":
    unittest.main()
