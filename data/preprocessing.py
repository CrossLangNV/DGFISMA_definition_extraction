from typing import List

import nltk


def tokenize(sentences, option='nltk'):
    """For experimenting with different tokenization options:
        * TODO BERT tokenizer
        * TODO Spacy
        * https://www.nltk.org/

    Args:
        sentences: list of sentence strings.
        option: If option not within available options it will raise an error

    Returns:
        list with list of sentence tokens.
    """

    tokenizers = {'nltk': nltk_tokenizer}

    return tokenizers[option](sentences)


def nltk_tokenizer(sentences: (str, List[str])) -> ([List[str]], List[List[str]]):
    """ tokenizer based from https://www.nltk.org/

    Args:
        sentences: can be either list of sentences or single sentence

    Returns:

    """
    if isinstance(sentences, list):

        sentences_tok = []
        for sentence in sentences:
            sentences_tok.append(' '.join(nltk.word_tokenize(sentence)))

    else:
        sentences_tok = nltk.word_tokenize(sentences)

    return sentences_tok
