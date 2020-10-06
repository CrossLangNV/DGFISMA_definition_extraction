from typing import List, Tuple, Set

from cassis import Cas


def get_text_html(cas: Cas, SofaID: str, tagnames: Set[str] = set('p')) -> (List[str], List[Tuple[int, int]]):
    '''
    Given a cas, and a view (SofaID), this function selects all ValueBetweenTagType elements ( with tag.tagName in the set tagnames ), extracts the covered text, and returns the list of extracted sentences and a list of Tuples containing begin and end posistion of the extracted sentence in the sofa.
    '''

    sentences = []
    begin_end_position = []
    for tag in cas.get_view(SofaID).select("com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"):
        if tag.tagName in set(tagnames):
            sentence = tag.get_covered_text().strip()
            if sentence:
                sentences.append(sentence)
                begin_end_position.append((tag.begin, tag.end))

    return sentences, begin_end_position


def get_text_pdf(cas: Cas, SofaID: str) -> (List[str], List[Tuple[int, int]]):
    '''
    Given a cas, and a view, this function should return the text we want.
    # TODO
    '''

    return ['test sentence'], [(0, 1)]
