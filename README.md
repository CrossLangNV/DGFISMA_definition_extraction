Instructions
------------

use "dbuild.sh" to build the docker image <br />
use "dcli.sh" to start a docker container

Given a document (json), e.g.: https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/example.json, the program will return a json containing **definitions**, e.g:

<em>
{
    "<strong>definitions</strong>": ‘financial customer’ means a customer that performs one or more of the activities listed in Annex I to Directive 2013/36/EU as its main business, or is one of the following:■■■■(a)■■■■a credit institution;■■■■(b)■■■■an investment firm;■■■■(c)■■■■a financial institution;■■■■(d)■■■■a securitisation special purpose vehicle (‘SSPE’);■■■■(e)■■■■a collective investment undertaking (‘CIU’);■■■■(f)■■■■a non-open ended investment scheme;■■■■(g)■■■■an insurance undertaking;■■■■(h)■■■■a reinsurance undertaking;■■■■(i)■■■■a financial holding company or mixed-financial holding company;\n‘personal investment company’ (‘PIC’) means an undertaking or a trust whose owner or beneficial owner, respectively, is a natural person or a group of closely related natural persons, which was set up with the sole purpose of managing the wealth of the owners and which does not carry out any other commercial, industrial or professional activity. The purpose of the PIC may include other ancillary activities such as segregating the owners' assets from corporate assets, facilitating the transmission of assets within a family or preventing a split of the assets after the death of a member of the family, provided these are connected to the main purpose of managing the owners' wealth;\n‘stress’ shall mean a sudden or severe deterioration in the solvency or liquidity position of a credit institution due to changes in market conditions or idiosyncratic factors as a result of which there may be a significant risk that the credit institution becomes unable to meet its commitments as they fall due within the next 30 calendar days;\n‘margin loans’ means collateralised loans extended to customers for the purpose of taking leveraged trading positions.
}
</em>

<br />
<br />
Each line in the json is a definition and its context. E.g.: <em>
  ‘financial customer’ means a customer that performs one or more of the activities listed in Annex I to Directive 2013/36/EU as its main business, or is one of the following:■■■■(a)■■■■a credit institution;■■■■(b)■■■■an investment firm;■■■■(c)■■■■a financial institution;■■■■(d)■■■■a securitisation special purpose vehicle (‘SSPE’);■■■■(e)■■■■a collective investment undertaking (‘CIU’);■■■■(f)■■■■a non-open ended investment scheme;■■■■(g)■■■■an insurance undertaking;■■■■(h)■■■■a reinsurance undertaking;■■■■(i)■■■■a financial holding company or mixed-financial holding company;</em>. 
<br />
<br />


First part, e.g.:
<em> ‘financial customer’ means a customer that performs one or more of the activities listed in Annex I to Directive 2013/36/EU as its main business, or is one of the following:</em>, was labeled by the classifier as being a definition. The context was then added ( using ■■■■ ) to this detected definition via regular expressions ( see: https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/annotate.py ). 


By default a Bert sequence classifier will be used for classification. Such a model can be trained via https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/bert_classifier/src/train.py

However, fastText sequence classifiers are also supported by https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/definition.py.

Make sure to update the path to the directory where the newly trained model is located in: https://github.com/ArneDefauw/DGFISMA/blob/master/definition_extraction/dbuild.sh. 

When using fastText models for classification, make sure to update the <em>Dockerfile</em> and  <em>app.py</em>

Note that models are not included in the repository, because they are too large. 
