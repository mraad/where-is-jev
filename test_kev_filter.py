import unittest
from experiments.kev_filter import partition, select_threshold

class FilterTests(unittest.TestCase):
    def test_split_excludes_crossing_crops(self):
        self.assertEqual(partition(10,40,[0,50,75,100]),"train")
        self.assertEqual(partition(45,55,[0,50,75,100]),"buffer")
        self.assertEqual(partition(80,95,[0,50,75,100]),"test")

    def test_threshold_preserves_calibration_positives(self):
        self.assertEqual(select_threshold([.1,.6,.9],[False,True,True]),.6)
        with self.assertRaises(ValueError):
            select_threshold([.1],[False])

if __name__=="__main__":
    unittest.main()
