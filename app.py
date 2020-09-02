#!/usr/local/bin/python
from flask import Flask
from flask import request
from flask import abort
import base64
import binascii
   
from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
    
from cleaning import get_text_html, get_text_pdf
from definition import DefinitionFinder

app = Flask(__name__)
   
TYPESYSTEM_PATH="/work/typesystems/typesystem.xml"
MODEL_PATH="/work/models/model.pth"
DEVICE='cpu'  # 'cpu', 'cuda:0', 'cuda:1',...
NR_OF_THREADS=12  #ignored when device is not equal to 'cpu'

@app.route('/extract_definitions', methods=['POST'])
def extract_definitions():    
    if not request.json:
        abort(400) 
    output_json={}
    
    if ('cas_content' not in request.json) or ( 'content_type' not in request.json ):
        print( "'cas_content' and/or 'content type' field missing" )
        output_json['cas_content']=''
        output_json['content_type']=''
        return output_json
        
    try:
        decoded_cas_content=base64.b64decode( request.json[ 'cas_content' ] ).decode( 'utf-8' )
    except binascii.Error:
        print( f"could not decode the 'cas_content' field. Make sure it is in base64 encoding." )
        output_json['cas_content']=''
        output_json['content_type']=request.json[ 'content_type' ]
        return output_json

    with open( TYPESYSTEM_PATH , 'rb') as f:
        typesystem = load_typesystem(f)

    #load the cas:
    cas=load_cas_from_xmi( decoded_cas_content, typesystem=typesystem  )

    if request.json[ 'content_type'] == 'pdf':

        sentences, begin_end_positions=get_text_pdf( cas , "html2textView" )
        
    elif request.json[ 'content_type'] == 'html' or request.json[ 'content_type'] == 'xhtml':

        sentences, begin_end_positions=get_text_html( cas , "html2textView" )

    else:
        print( f"content type { request.json[ 'content_type'] } not supported by paragraph annotation app" )   
        output_json['cas_content']=request.json['cas_content']
        output_json['content_type']=request.json[ 'content_type' ]
        return output_json   
    
    definition_finder=DefinitionFinder( sentences  )

    #definitions=definition_finder.get_definitions_regex()
    definitions=definition_finder.get_definitions_bert( model_path=MODEL_PATH, device=DEVICE )
    
    #sanity check
    assert len(definitions) == len(sentences) == len( begin_end_positions )

    Sentence=typesystem.get_type( 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence' )

    #Only annotate sentences that are definitions
    for definition, begin_end_position in zip( definitions, begin_end_positions ):
        if definition:
            cas.get_view( 'html2textView' ).add_annotation( Sentence(begin=begin_end_position[0] , end=begin_end_position[1] , id="definition"  ) )  

    output_json['cas_content']=base64.b64encode(  bytes( cas.to_xmi()  , 'utf-8' ) ).decode()   
    output_json[ 'content_type']=request.json[ 'content_type']
        
    return output_json

    
@app.route('/')
def index():
    return "Up and running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)