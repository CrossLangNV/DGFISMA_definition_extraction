import os
import unittest

from retraining.data import get_gold_standard, read_lines


class TestData(unittest.TestCase):

    def test_read_lines(self):
        DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../test_files/arne')
        DATA_FOLDER = os.path.normpath(DATA_FOLDER)
        path_file = os.path.join(DATA_FOLDER, 'test_sentences')

        l = read_lines(path_file)

        self.assertIsInstance(l, list, 'Should return a list')
        for s in l:
            self.assertIsInstance(s, str, 'Returned list should contain strings')

    def test_get_gold_standard(self):
        data = get_gold_standard()

        self.assertEqual(2, len(data), "Should return two values: x and y")

        x, y = data

        self.assertTrue(x, 'Should return a non-empty x')
        self.assertTrue(y, 'Should return a non-empty y')

        self.assertEqual(len(x), len(y), 'input and output should be of equal length')

        self.assertIsInstance(x, list, 'x should be a list')
        for s in x:
            self.assertIsInstance(s, str, 'x should contain sentences as strings')

        self.assertIsInstance(x, list, 'y should be a list')
        for i in y:
            self.assertIsInstance(i, int, 'y should contain indices as integers')


if __name__ == '__main__':
    unittest.main()
