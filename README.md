Instructions
------------

use "dbuild.sh" to build the docker image <br />
use "dcli.sh" to start a docker container

Don't forget to:

1) Set the path to the directory where the BERT model for classification is located in `dbuild.sh` . 
(See https://github.com/CrossLangNV/DGFISMA_definition_extraction/releases/tag/v0.0.1 for such a trained (Distil)BERT model for classification)

2) Set the path to the correct typesystem in `dbuild.sh`.

Given a json, e.g.: https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/tests/test_files/json/small_nested_tables.json , with a "cas_content" and "content_type" field, a json with the same fields will be returned (e.g. https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/tests/test_files/response_json/small_nested_tables_response.json), but with definition annotations added. 

The "cas_content" is a UIMA CAS object, encoded in base64. The "content_type" can be "html" or "pdf". 

For working with a CAS object in python, [the dkpro-cassis library](https://github.com/dkpro/dkpro-cassis) is used.

## Definition extraction

The task of definition extraction/detection is approached as a sentence classification task. A pre-trained DistilBert Model (provided by https://github.com/huggingface/transformers) with a Sequence classification layer (DistilBertSequenceClassifier) on top is used for sentence classification. Such a model can be trained via the provided user scripts (see below).

A trained DistilBertSequenceClassifier + training data and test data can be found here: https://github.com/CrossLangNV/DGFISMA_definition_extraction/releases/tag/v0.0.1

The algorithm consists of the following steps:

1. Retrieve text segments:
    
    We select the view `html2textView` of the CAS object, and retrieve all `com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType` features with tagname equal to `p`. This results in a list of sentences (list of strings), and their offsets.
    
2. Sentence classification:

    The resulting list of sentences is classified by the DistilBertSequenceClassifier.
    
3. Annotation of definitions:

    If a sentence was labeled as a definition, we use the offset, obtained in step 1, to add a `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` annotation to the `html2textView` view of the CAD, with `id="definition"`. Sentences that are labeled as not containing a definition, do not receive an annotation.

4. Add lists/sublists/enumeration:

    To add context (i.e. lists/sublists/enumeration) to the detected definitions, we use a paragraph annotation `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph`, that can be added to the `html2textView` view of the CAS using the paragraph annotation app: https://github.com/CrossLangNV/DGFISMA_paragraph_detection .

Note that models for sentence classification are not included in the repository, because they are too large. 

## User scripts

### Retraining

To enable retraining, user scripts are provided to make it easy to train a new model, evaluate it and export to a folder.

These user_scripts are provided in:
* [user_scripts/classifier_train.py](user_scripts/classifier_train.py)
* [user_scripts/classifier_eval.py](user_scripts/classifier_eval.py)
* [user_scripts/classifier_pred.py](user_scripts/classifier_pred.py)

An example of a whole retraining worklow is proved in:
* [examples/example_retraining_flow.py](examples/example_retraining_flow.py) 

### Evaluation of DistilBertSequenceClassifier and a supervised FastText model for text Classification: 

To compare the performance of a DistilBertSequenceClassifier to FastText, we provide some user scripts:

* [examples/train_bert_main.py](examples/train_bert_main.py)
* [examples/train_fasttext_main.py](examples/train_fasttext_main.py)
* [examples/performance_main.py](examples/performance_main.py) for plotting the distribution of the performance measures. Folder with generated logs can be provided and is evaluated.

To compare the actual inference speed, a seperate script can be found at:
* [examples/evaluate_inference.py](examples/evaluate_inference.py)