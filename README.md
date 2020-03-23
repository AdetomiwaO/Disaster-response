# Disaster Response Pipeline Project

### Project Description:
- The goal of the project was to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data sets include a disaster message data set along with category labels for 36 disaster categories. 

### Project Components
- ETL Pipeline
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

- Machine Learning Pipeline
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
- Flask Web App
Web app that receives a new message and outputs category classifications using the trained model 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
