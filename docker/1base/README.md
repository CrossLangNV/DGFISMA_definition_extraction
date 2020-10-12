`
docker build
--build-arg MODEL_DIR=bert_classifier/models_dgfisma_def_extraction/run_2020_06_26_11_56_31_acb319aac70b
--build-arg TYPESYSTEM_PATH=tests/test_files/typesystems/typesystem.xml
-t definition_extraction_app/base .
`