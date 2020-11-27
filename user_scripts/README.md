# pre-trained BERT classifier

In the context of DGFISMA a pre-trained BERT classifier is trained for classifying sentence being definitions.

## Training

To allow updating the model, the following function can be called or used as a user script.

    from user_scripts.classifier_train.py import main
    
    train_sentences = "train_sentences.txt"
    train_labels = "test_labels.txt"
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

## Evaluation