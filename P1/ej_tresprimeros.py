from nltk.util import ngrams
import regex as re

def tokenize_by_spaces(text):
    return text.split()


def tokenize_by_punctuation(text):
    pattern = r"\w+|[^\w\s]|\p{So}"
    tokens = re.findall(pattern, text, flags=re.UNICODE)
    return tokens

def tokenize_by_ngrams(text, n):

    words = tokenize_by_spaces(text)
    return list(ngrams(words, n))
