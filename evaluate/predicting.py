import fasttext

from bert_classifier.src.predict import main


def predict_bert(filename: str,
                 model_path: str,
                 output_file: str):
    main(['--filename', filename,
          '--model_path', model_path,
          '--output_file', output_file])

    pred = _open_lines(output_file)

    pred_argmax = [int(a.split()[0]) for a in pred]

    return pred_argmax


def predict_fasttext(model_path, sentences):
    model = fasttext.load_model(
        model_path)

    # sanity check:
    assert (set(model.labels) == set(["__label__no_definition", "__label__definition"]))

    pred_labels = model.predict(sentences)[0]
    pred_labels = [1 if label[0] == "__label__definition" else 0 for label in
                   pred_labels]  # explicitely use the labels

    return pred_labels


def _open_lines(path):
    with open(path) as f:
        return f.read().strip("\n").split("\n")
