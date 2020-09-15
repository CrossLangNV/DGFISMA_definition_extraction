import os
import tempfile
import unittest


from evaluate.predicting import predict_fasttext, predict_bert, Prediction

# to import files from base folder.
os.chdir("../..")

path_file = 'tests/test_files/arne/test_sentences'


def _open_lines(path):
    with open(path) as f:
        return f.read().strip("\n").split("\n")


TEST_SENTENCES = _open_lines(path_file)

BERT_model_path = "bert_classifier/models_dgfisma_def_extraction/run_2020_06_26_11_56_31_acb319aac70b/distilbert-base-uncased_model_10.pth"


class TestPrediction(unittest.TestCase):
    def test_predict_bert(self):

        with tempfile.NamedTemporaryFile() as temp:
            path_out = temp.name

            pred = predict_bert(path_file, BERT_model_path, path_out)

            self.assertTrue(os.path.exists(path_out), 'No file found')

        self.assertEqual(len(TEST_SENTENCES), len(pred), 'Every sentence should have a prediction')

        self.assertTrue(_test_pred(pred), 'Output should be list of indices as integers')

        self.assertFalse(os.path.exists(path_out), 'File should only be temporary')

    def test_predict_bert_without_saving(self):
        pred = predict_bert(path_file, BERT_model_path)

        self.assertIsInstance(pred, Prediction)
        self.assertTrue(pred)

    def test_fasttext(self):

        model_path = 'tests/test_files/arne/model_fasttext.bin'

        pred_labels = predict_fasttext(model_path, TEST_SENTENCES)

        self.assertTrue(pred_labels, 'Output should be non-empty')

        self.assertEqual(len(TEST_SENTENCES), len(pred_labels), 'Amount of predictions should equal inputs')

        self.assertTrue(_test_pred(pred_labels), 'Output should be list of indices as integers')


def _test_pred(pred):

    return all([isinstance(a, int) for a in pred])



if __name__ == '__main__':
    unittest.main()
