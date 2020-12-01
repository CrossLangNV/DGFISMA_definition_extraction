import os
import tempfile
import unittest

from data.load import SentenceData, LabelData
from user_scripts import classifier_eval, classifier_pred, classifier_train


class TestRetrainingFlow(unittest.TestCase):
    """Tests for retraining
    """

    def test_retraining_flow(self):
        """Test the whole retraining flow

        Returns:
            None
        """
        sentences = SentenceData.from_file()
        labels = LabelData.from_file()

        s_train, l_train = zip(*list(zip(sentences, labels))[::2])
        s_test, l_test = zip(*list(zip(sentences, labels))[1::2])

        with tempfile.NamedTemporaryFile(suffix='.txt') as f_x_train, \
                tempfile.NamedTemporaryFile(suffix='.txt') as f_y_train, \
                tempfile.NamedTemporaryFile(suffix='.txt') as f_x_test, \
                tempfile.NamedTemporaryFile(suffix='.txt') as f_y_test, \
                tempfile.TemporaryDirectory() as d_model, \
                tempfile.NamedTemporaryFile(suffix='.txt') as f_pred_test:

            with open(f_x_train.name, 'w+') as f:
                for s_i in s_train:
                    f.write(f'{s_i}\n')

            with open(f_y_train.name, 'w+') as f:
                for l_i in l_train:
                    f.write(f'{l_i}\n')

            with open(f_x_test.name, 'w+') as f:
                for s_i in s_test:
                    f.write(f'{s_i}\n')

            with open(f_y_test.name, 'w+') as f:
                for l_i in l_test:
                    f.write(f'{l_i}\n')

            classifier_train.main(f_x_train.name,
                                  f_y_train.name,
                                  d_model, epochs=1)

            model_folder = sorted(f for f in os.listdir(d_model) if os.path.isdir(os.path.join(d_model, f)))[-1]
            model_dir = os.path.join(d_model, model_folder)

            classifier_pred.main(model_dir,
                                 f_x_test.name,
                                 f_pred_test.name)

            classifier_eval.main(f_y_test.name,
                                 f_pred_test.name,
                                 model_dir)
