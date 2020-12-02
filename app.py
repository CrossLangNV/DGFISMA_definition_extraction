#!/usr/local/bin/python
import base64
import binascii
import logging
import os

from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from flask import Flask
from flask import abort
from flask import request

from utils import get_sentences
from models.preconfigured import BERTForDefinitionClassification

app = Flask(__name__)

PATH_TYPESYSTEM = "/work/typesystems/typesystem.xml"
PATH_MODEL = "/work/models"
DEVICE = os.getenv('DEVICE', 'cpu')  # 'cpu', 'cuda:0', 'cuda:1',...
NR_OF_THREADS = 12  # ignored when device is not equal to 'cpu'

print(f'Using device: {DEVICE}')
logging.info(f'Using device: {DEVICE}')

MODEL = BERTForDefinitionClassification.from_dir(PATH_MODEL, device=DEVICE)

SOFA_ID="html2textView"
VALUE_BETWEEN_TAG_TYPE="com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"
TAG_NAMES="p"
DEFINITION_TYPE="de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

@app.route('/extract_definitions', methods=['POST'])
def extract_definitions():
    if not request.json:
        abort(400)
    output_json = {}

    if ('cas_content' not in request.json) or ('content_type' not in request.json):
        print("'cas_content' and/or 'content type' field missing")
        output_json['cas_content'] = ''
        output_json['content_type'] = ''
        return output_json

    try:
        decoded_cas_content = base64.b64decode(request.json['cas_content']).decode('utf-8')
    except binascii.Error:
        logging.exception(f"could not decode the 'cas_content' field. Make sure it is in base64 encoding.")
        output_json['cas_content'] = ''
        output_json['content_type'] = request.json['content_type']
        return output_json

    with open(PATH_TYPESYSTEM, 'rb') as f:
        typesystem = load_typesystem(f)

    # load the cas:
    cas = load_cas_from_xmi(decoded_cas_content, typesystem=typesystem)

    try:
        sentences, begin_end_positions = get_sentences(cas, SOFA_ID, tagnames=set( TAG_NAMES ), value_between_tagtype=VALUE_BETWEEN_TAG_TYPE  )
    except:
        logging.exception(f"Could not extract sentences/offsets from the cas object. This is unwanted behaviour.")
        output_json['cas_content'] = request.json['cas_content']
        output_json['content_type'] = request.json['content_type']
        return output_json

    definitions, _ = MODEL.predict(sentences)

    # sanity check
    assert len(definitions) == len(sentences) == len(begin_end_positions)

    SentenceClass = typesystem.get_type( DEFINITION_TYPE )

    # Only annotate sentences that are definitions
    for definition, begin_end_position in zip(definitions, begin_end_positions):
        if definition:
            cas.get_view(SOFA_ID).add_annotation(
                SentenceClass(begin=begin_end_position[0], end=begin_end_position[1], id="definition"))

    output_json['cas_content'] = base64.b64encode(bytes(cas.to_xmi(), 'utf-8')).decode()
    output_json['content_type'] = request.json['content_type']

    return output_json


@app.route('/')
def index():
    return "Up and running"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
