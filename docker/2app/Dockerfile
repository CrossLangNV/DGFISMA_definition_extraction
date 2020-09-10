FROM definition_extraction_app/base

ARG MODEL_DIR
ARG TYPESYSTEM_PATH

WORKDIR /work
#copy code:
COPY app.py /work
COPY cleaning.py /work
COPY definition.py /work
COPY bert_classifier/src/models.py /work/bert_classifier/src/

#copy model:
COPY $MODEL_DIR/*.pth /work/models/model.pth
COPY $MODEL_DIR/config.json /work/models/
COPY $MODEL_DIR/special_tokens_map.json /work/models/
COPY $MODEL_DIR/tokenizer_config.json /work/models/
COPY $MODEL_DIR/vocab.txt /work/models/

#copy typesystem:
COPY $TYPESYSTEM_PATH /work/typesystems/typesystem.xml

CMD python /work/app.py