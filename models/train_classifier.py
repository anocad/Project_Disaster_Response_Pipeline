import sys
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sqlalchemy import create_engine
import sqlite3
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.svm import LinearSVC

import pickle
import warnings
warnings.filterwarnings('ignore') 



def load_data(db_path):
    """
    Loads disaster data from SQLite database
    
    Parameters:
        database_filepath(str): path to the SQLite database
        table_name(str): name of the table where the data is stored
        
    Returns:
        (DataFrame) X: Independent Variables , array which contains the text messages
        (DataFrame) Y: Dependent Variables , array which contains the labels to the messages
        (DataFrame) category: Data Column Labels , a list with the target column names, i.e. the category names
    """
    engine = create_engine('sqlite:///' + db_path)
    table_name = 'disaster_response_table'
    df = pd.read_sql_table(table_name, engine)

    X = df['message']
    y = df.iloc[:,5:]

    categories = y.columns

    return X, y, categories 



def tokenize(text):
    """
    Tokenizes message data
    
    INPUT:
       text (string): message text data
    
    OUTPUT:
        (DataFrame) clean_messages: array of tokenized message data
    """
    regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # apply regex to get urls
    found_urls = re.findall(regex, text)
    
    # replace urls with placeholder
    for url in found_urls:
        text = text.replace(url,'placeholder')
    
    lemmatizer = WordNetLemmatizer()

    # confert text into tokens
    tokens = word_tokenize(text)
    
    # go through each token and do the following: lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens
'''
def build_model():
    """
    This function output is a SciKit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
        
    Returns:
        gs_cv(obj): an estimator which chains together a nlp-pipeline with
                    a multi-class classifier
    
    """
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200,random_state=20)))
    ])
    
    # Only limited number of paramters is specified, since the more training was done in jupyter notebook
    parameters = {
        'clf__estimator__max_depth': [2, None]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv
'''

def build_model():
    pipeline = Pipeline([
        #('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameters set to this due to reduce the size of pkl file, which were too large (600MB) for uploading to github with my previous parameters.
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model

def evaluate_model(model, X_test, Y_test, categories):

    """
    Evaluates models performance in predicting message categories
    
    INPUT:
        model (Classification): stored classification model
        X_test (string): Independent Variables
        Y_test (string): Dependent Variables
        categories (DataFrame): Stores message category labels
    OUTPUT:
        None. But prints a classification report for every category
    """

    Y_pred = model.predict(X_test)

    # Results of the Results of the classification report::
    print("Results of the classification report:")
    print(classification_report(Y_test, Y_pred, target_names=categories))


def save_model(model, model_filepath):
    
    """
    Saves trained classification model to pickle file
    
    INPUT:
        model (Classification): stored classification model
        model_filepath (string): Filepath to pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'),-1)


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
        
    Args:
        None
        
    Returns:
        Nothing
 
    """    
    
    if len(sys.argv) == 3:
        db_path, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(db_path))
        X, Y, categories = load_data(db_path)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

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