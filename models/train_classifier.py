# import packages
import sys
import sqlalchemy as db
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize, punkt
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    # read in file
    engine = db.create_engine('sqlite:///{}'.format(database_filepath))
    connection = engine.connect()
    metadata = db.MetaData()
    
    df = pd.read_sql_table('DisasterResponse_db', connection)
    # define features and label arrays
    X = df.message
    y = df.drop(['id','message','original','genre'], axis = 1)
    category_names = list(y.columns)
    
    return X, y, category_names

def tokenize(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [item.replace(r'[^\w\s]',' ') for item in tokens]
    tokens = (' '.join([item for item in tokens if item not in stop_words]))
    tokens = [lemmatizer.lemmatize(x) for x in tokens]
    
    return tokens


def build_model():
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf))
    ])    

    # define parameters for GridSearchCV
    parameters = {'clf__estimator__min_samples_split': [2, 3, 4],
                  'vect__ngram_range': ((1, 2), (2,2))
                 }

    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
 
    print("\nBest Parameters:", model.best_params_)
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = category_names)
    
    for col in category_names: 
        result = classification_report(Y_test[col], y_pred_df[col])
    print(col, result)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        #edit the file
        pickle.dump(model, f)
        #close file
        f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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