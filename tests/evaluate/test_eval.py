import os
import unittest

from evaluate.eval import *
from data.load import get_gold_standard, read_lines

# to import files from base folder.
BASE_FOLDER = os.path.join(os.path.dirname(__file__), "../..")


class TestEvaluation(unittest.TestCase):
    def test_report(self):
        path_predict = os.path.join(BASE_FOLDER, 'tests/test_files/arne/test_sentences_predict')

        test_sentences_predict = read_lines(path_predict)
        test_sentences_predict = [int(label.split()[0]) for label in test_sentences_predict]

        _, test_labels = get_gold_standard()

        report_str = report(test_labels, test_sentences_predict)

        self.assertTrue(report_str, 'report should be non-empty')


if __name__ == '__main__':
    unittest.main()
