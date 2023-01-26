import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
    

class Preprocess(BaseEstimator, TransformerMixin):
  
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        def tokenize(text):
            #remove non-alphanumeric characters
            text = re.sub(r"[^a-zA-Z0-9]", " ", text)
            
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()

            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)

            return ' '.join(clean_tokens)
      

        return pd.Series(X).apply(tokenize).values