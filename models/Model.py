import sys
import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from Utils import tokenize
import joblib

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Data', engine)
    X = df.message.apply(tokenize)
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameters set to this due to reduce the size of pkl file, which were too large (840 MB) for uploading to github with my previous parameters.
    parameters = {
        'clf__estimator__n_estimators': [8],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=2, cv=4)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    print(class_report)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        joblib.dump(model, file)


def main():
    print(sys.argv)
    print(len(sys.argv))
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()