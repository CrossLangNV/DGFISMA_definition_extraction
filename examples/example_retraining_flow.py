import os
import tempfile

from user_scripts import classifier_eval, classifier_pred, classifier_train

if __name__ == '__main__':
    """
    Script that does all the important steps for retraining:
    1. Training a new model on new data
    2. Evaluating the new model
    3. Producing new predictions
    """

    """
    Settings
    """
    EXAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "output")

    ROOT = os.path.join(os.path.dirname(__file__), '..')

    path_train_x = os.path.join(ROOT, 'tests/test_files/arne/train_dgfisma_wiki_sentences.txt')
    path_train_y = os.path.join(ROOT, 'tests/test_files/arne/train_dgfisma_wiki_labels.txt')

    path_test_x = os.path.join(ROOT, 'tests/test_files/arne/test_sentences')
    path_test_y = os.path.join(ROOT, 'tests/test_files/arne/test_labels')

    b_train = True
    b_evaluate = True
    b_predict = True

    """
    The 3 components of workflow
    """
    if b_train:
        """
        Train a model (based on some new data)
        """

        path_model = EXAMPLE_FOLDER

        classifier_train.main(path_train_x,
                              path_train_y,
                              path_model)

    if b_evaluate:
        """
        Performance results are saved in <EXAMPLE_FOLDER>/log.txt 
        """

        model_folder = sorted(f for f in os.listdir(EXAMPLE_FOLDER) if os.path.isdir(os.path.join(EXAMPLE_FOLDER, f)))[
            -1]
        # Evaluate this model

        model_dir = os.path.join(EXAMPLE_FOLDER, model_folder)

        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            classifier_pred.main(model_dir,
                                 path_test_x,
                                 f.name)

            classifier_eval.main(path_test_y,
                                 f.name,
                                 EXAMPLE_FOLDER)

    if b_predict:
        """
        Produce predictions for whole dataset
        """
        model_folder = sorted(f for f in os.listdir(EXAMPLE_FOLDER) if os.path.isdir(os.path.join(EXAMPLE_FOLDER, f)))[
            -1]

        model_dir = os.path.join(EXAMPLE_FOLDER, model_folder)
        # If it is better, based on evaluation, select it and update predictions

        filename_pred = os.path.join(EXAMPLE_FOLDER, 'pred.txt')
        classifier_pred.main(model_dir,
                             path_train_x,
                             filename_pred)

    print("finished")
