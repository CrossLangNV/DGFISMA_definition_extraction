docker build \
--no-cache \
--build-arg MODEL_DIR=bert_classifier/models_dgfisma_def_extraction/retraining_example \
--build-arg TYPESYSTEM_PATH=tests/test_files/typesystems/typesystem.xml \
-f docker/2app/Dockerfile \
-t definition_extraction_app/retraining .
