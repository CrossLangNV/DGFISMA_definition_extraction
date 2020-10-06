import tempfile

import fasttext

from bert_classifier.src.predict import main


class Prediction(list):
    def __init__(self, l):

        for i in l:
            assert isinstance(i, int)

        return super(Prediction, self).__init__(l)


def predict_bert(filename: str,
                 model_path: str,
                 output_file: str = None
                 ) -> Prediction:

    if output_file is None:
        with tempfile.NamedTemporaryFile() as temp:
            output_file = temp.name

            main(['--filename', filename,
                  '--model_path', model_path,
                  '--output_file', output_file])

            pred = _open_lines(output_file)

    else:

        main(['--filename', filename,
              '--model_path', model_path,
              '--output_file', output_file])

        pred = _open_lines(output_file)

    pred_argmax = [int(a.split()[0]) for a in pred]

    return Prediction(pred_argmax)


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
