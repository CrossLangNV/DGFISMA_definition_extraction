import os
import unittest

from data.load import get_gold_standard, read_lines, get_training_data, DefinitionData


class TestDefinitionData(unittest.TestCase):

    def test_create(self):
        sentences = ['one', 'two']
        labels = [1, 2]

        self.assertEqual((sentences, labels), DefinitionData(sentences, labels))

    def test_wrong_input(self):
        with self.assertRaises(Exception, msg='sentences should contain strings') as context:
            sentences = [1, 'two']
            labels = [1, 2]

            DefinitionData(sentences, labels)

        with self.assertRaises(Exception, msg='sentences should contain strings'):
            sentences = ['one', 'two']
            labels = [1, 'two']

            DefinitionData(sentences, labels)

        with self.assertRaises(Exception, msg='Equal length required'):
            sentences = ['one']
            labels = [1, 2]

            DefinitionData(sentences, labels)


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

        self._validation_data(data)

    def test_get_training_data(self):

        data = get_training_data()

        self._validation_data(data)

    def _validation_data(self, data):
        """
        Every output should be of same type
        """

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
