# BERT for Sequence Classification

In the context of DGFISMA, a sequence classifier is used to detemine if a sentence is a definition (of a certain concept). For this classification task a pre-trained DistilBert Model (provided by https://github.com/huggingface/transformers) with a Sequence classification layer (DistilBertSequenceClassifier) on top is used.

### Example:

Given two sentences:

```
For the purpose of this paragraph annual contribution means the contribution to be collected by the Board for a given financial year in accordance with this Regulation in order to cover the administrative expenditures of the Board;
```

and 

```
The liabilities referred to in paragraph 1(a) and (b) shall be evenly deducted on a transaction by transaction basis from the amount of total liabilities of the institutions which are parties of the transactions or agreements referred to in paragraph 1(a) and (b).
```
, our trained DistilBertSequenceClassifier should only label the first one as a definition. A Bert model for Token Classification ( see https://github.com/CrossLangNV/DGFISMA_term_extraction ), is then used to determine which terms are defined by the detected definition. 

## Training

To allow updating the model, the following function can be called or used as a user script.

    from user_scripts.classifier_train import main
    
    train_sentences = "train_sentences.txt"
    train_labels = "train_labels.txt"
    model_storage_directory = "models/model1"
    main(train_sentences,
         train_labels,
         model_storage_directory)
    
With 
* `train_sentences`: path to file with training sentences. Each line being a sentence. An example is provided at [file](../tests/test_files/arne/test_sentences)
* `train_labels`: path to file with training labels. Should match with `train_sentences`. Each line containing a 1 if the respective sentence is a definition, else 0. An example is provided at [file](../tests/test_files/arne/test_labels)
* `model_storage_directory`: path to folder where model is saved. 

It can also be directly called as a user-script:

    python user_scripts/classifier_train.py "train_sentences.txt" "test_labels.txt" "models/model1"
    
Extra flags are:
   
* `-h`: provide help.
* `-validation <integer>`: Default .2. Fraction of data to use as validation.
* `-validation-sentences <path>`]: Default None. Path to validation dataset. If provided, `-validation` fraction will be ignored.
* `-epochs <interger>`: Default 10, number of epochs to train the model.

## Inference

The trained model can be used to infer a file with new sentences.

    from user_scripts.classifier_pred import main
 
    model_dir = "models/model1"
    path_x = "test_sentences.txt"
    path_pred = "test_predictions.txt"
    main(model_dir,
             path_x,
             path_pred)
             
With 
* `model_dir`: Directory containing the model (see [classifier_train.py](./classifier_train.py))
* `path_x`: Filename with sentences to be classified
* `path_pred`: Optional. Filename of output of where predictions are saved

The output is saved as `<label index> [<likelihood no defintion>, <likelihood definition>]`.

It can also be directly called as a user-script:

    python user_scripts/classifier_pred.py "models/model1" "test_sentences.txt" "test_predictions.txt"

## Evaluation

If a ground truth is available, the prediction can be compared and evaluated on some metrics like precision, recall and f1-score.

    from user_scripts.classifier_eval import main
 
    path_y = "test_labels.txt"
    path_pred = "test_predictions.txt"
    path_logger = "."
    main(path_y,
             path_pred,
             path_logger=path_logger)
             
With 
* `path_y`: Filename of the labels. 
* `path_pred`: Filename of the prediction results (see [classifier_pred.py](./classifier_pred.py))
* `path_logger`: Filename where scores are saved as a CSV.

It can also be directly called as a user-script:

    python user_scripts/classifier_eval.py "test_labels.txt" "test_predictions.txt" -path-logger "."
   