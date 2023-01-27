from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


def tokenize(text):
      #remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return " ".join(clean_tokens)