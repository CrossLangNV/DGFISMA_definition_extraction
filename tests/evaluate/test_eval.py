import os
import unittest

from evaluate.eval import *

# to import files from base folder.
os.chdir("../..")

class TestEvaluation(unittest.TestCase):
    def test_report(self):

        path_predict = 'tests/test_files/arne/test_sentences_predict'

        path_labels = 'tests/test_files/arne/test_labels'

        def open_lines(path):
            with open(path) as f:
                return f.read().strip("\n").split("\n")

        test_sentences_predict = open_lines(path_predict)
        test_sentences_predict = [int(label.split()[0]) for label in test_sentences_predict]

        test_labels = open_lines(path_labels)
        test_labels = list(map(int, test_labels))

        report_str = report(test_labels, test_sentences_predict)

        self.assertTrue(report_str, 'report should be non-empty')


if __name__ == '__main__':
    unittest.main()
