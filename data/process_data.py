import sys
import pandas as pd
import sqlalchemy as db

def load_data(messages_filepath, categories_filepath):
    '''
    load messages and category label dataframes
    merge dataframes together on the common key 'id'
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    
    return df

def clean_data(df):
    '''
    extract binary variables, clean dataframe, drop unwanted columns
    replace non-binary values, dro duplicate rows
    
    '''
    categories = df.categories.str.split(';' , expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    category_colnames = [x[0:(len(x)-2)] for x in row.loc[0]]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # replace numeric value of 2 with 1 in the related column
    categories['related'] = categories['related'].replace(2,1)
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis =1 )
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    # drop rows with blank classifications

    return df
    
    
def save_data(df, database_filepath):
    '''
    save cleaned dataframe to sqlite database
    
    '''
    engine = db.create_engine('sqlite:///{}'.format(database_filepath))                     
    df.to_sql('DisasterResponse_db', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()