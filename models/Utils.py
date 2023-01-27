from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


def tokenize(text):
    """
    This function remove non-alphanumeric characters then normalize and lemmatize words
    to return a clean text
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return " ".join(clean_tokens)
