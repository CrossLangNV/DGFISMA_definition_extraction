import tempfile

import fasttext

class Prediction(list):
    def __init__(self, l):

        for i in l:
            assert isinstance(i, int)

        return super(Prediction, self).__init__(l)

def predict_fasttext(model_path, sentences) -> Prediction:
    model = fasttext.load_model(
        model_path)

    # sanity check:
    assert (set(model.labels) == set(["__label__no_definition", "__label__definition"]))

    pred_labels = model.predict(sentences)[0]
    pred_labels = [1 if label[0] == "__label__definition" else 0 for label in
                   pred_labels]  # explicitely use the labels

    return Prediction(pred_labels)


def _open_lines(path):
    with open(path) as f:
        return f.read().strip("\n").split("\n")
