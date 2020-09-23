Instructions
------------

use "dbuild.sh" to build the docker image <br />
use "dcli.sh" to start a docker container

Don't forget to:

1) Set the path to the directory where the BERT model for classification is located in: https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/dbuild.sh. 

2) Set the path to the correct typesystem in dbuild.sh ( e.g. https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/dbuild.sh )

Given a json, e.g.: https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/tests/test_files/json/small_nested_tables.json , with a "cas_content" and "content_type" field, a json with the same fields will be returned (e.g. https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/tests/test_files/response_json/small_nested_tables_response.json), but with definition annotations added. 

The "cas_content" is a UIMA CAS object, encoded in base64. The "content_type" can be "html" or "pdf". 

If definitions are found, definition annotations will be added to the CAS object ( "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence" ) on the 'html2textView', with id="definition".

Sentences that are labeled as not containing a definition, do not receive an annotation.

By default a Bert sequence classifier will be used for classification. Such a model can be trained via https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/bert_classifier/src/train.py

However, fastText sequence classifiers are also supported by https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/master/definition.py .

When using fastText models for classification, make sure to update the <em>Dockerfile</em> and  <em>app.py</em>

Note that models for sentence classification are not included in the repository, because they are too large. 

To add context (i.e. lists/sublists) to the detected definitions, use the paragraph annotations "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph", that can be added to the 'html2textView' of the CAS using the paragraph annotation app: https://github.com/CrossLangNV/DGFISMA_paragraph_detection .

# Training and evaluation the model

In [user_scripts/train_bert_main.py](user_scripts/train_bert_main.py) multiple bert models are trained to be able to compare performances. 

In [user_scripts/evaluate_main.py](user_scripts/evaluate_main.py) the folder with logs can be inputted and evaluated.

# Retraining workflow

Whole retraining workflow example can be found at:
[examples/example_retraining_flow.py](examples/example_retraining_flow.py) 