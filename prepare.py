
import pandas as pd
from env import host, user, password
import os
from sklearn.model_selection import train_test_split

def get_url(db):
    '''
    This function takes in a database name and returns a url (using the specified 
    database name as well as host, user, and password from env.py) for use in the 
    pandas.read_sql() function.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_telco():
    '''
    This function first searches for a .csv file containing the telco data and then 
    reads that file into a dataframe named telco. If telco.csv is not found, it uses
    the get_url() helper function to access the SQL server and read the query result 
    into a dataframe named telco. It then caches this data into a csv file (telco.csv).
    This function takes no arguments and returns the telco dataframe.
    '''
    if os.path.isfile('telco.csv'):
        telco = pd.read_csv('telco.csv', index_col=0)
        return telco
    else:
        sql = '''
        SELECT *
        FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
        '''
        telco = pd.read_sql(sql, get_url('telco_churn'))
        telco.to_csv('telco.csv')
        return telco

def clean_telco(telco):
    '''
    This function takes in the telco dataframe and returns a clean version of the 
    dataframe. It drops unnecessary or duplicate columns and rows, prepares the data
    to work with machine learning models, and renames columns for clarity.
    '''
    telco = telco.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    telco = telco.drop_duplicates()
    telco = telco[telco.total_charges != ' ']
    telco.total_charges = telco.total_charges.astype(float)
    dummies = pd.get_dummies(telco[telco.select_dtypes('O').columns], drop_first=True)
    telco = pd.concat([telco, dummies], axis=1)
    telco.drop(columns=telco.select_dtypes('O').columns, inplace=True)
    telco.rename(columns={'gender_Male':'is_male',
                      'partner_Yes':'has_partner',
                      'dependents_Yes':'has_dependents',
                      'phone_service_Yes':'has_phone_service',
                      'paperless_billing_Yes':'has_paperless_billing',
                      'churn_Yes':'has_churned'}, inplace=True)
    telco['customer_id'] = acquire_telco()[acquire_telco().total_charges != ' '].customer_id
    return telco

def split_telco(telco):
    '''
    This function takes in the telco dataframe and splits it into three dataframes.
    It returns these dataframes in this order: train, validate, test.
    Train makes up 56% of the total observations, validate 24%, and test 20%.
    '''
    train, test = train_test_split(telco, test_size=0.2, random_state=123, stratify=telco.has_churned)
    train, validate = train_test_split(train, test_size=0.3, random_state=123, stratify=train.has_churned)
    return train, validate, test

def wrangle_telco():
    '''
    This function combines the acquire_telco(), clean_telco() and split_telco() functions.
    It takes no arguments, cleans the telco data, splits it, and returns three dataframes 
    called train, validate, and test.
    '''
    telco = acquire_telco()
    telco = clean_telco(telco)
    train, validate, test = split_telco(telco)
    return train, validate, test

def scale_telco(train, validate, test):
    mm_scaler = sklearn.preprocessing.MinMaxScaler()
    mm_scaler.fit(train[['total_charges', 'tenure', 'monthly_charges']])
    train[['total_charges_scaled', 'tenure_scaled', 'monthly_charges_scaled']] = mm_scaler.transform(train[['total_charges', 'tenure', 'monthly_charges']])
    validate[['total_charges_scaled', 'tenure_scaled', 'monthly_charges_scaled']] = mm_scaler.transform(validate[['total_charges', 'tenure', 'monthly_charges']])
    test[['total_charges_scaled', 'tenure_scaled', 'monthly_charges_scaled']] = mm_scaler.transform(test[['total_charges', 'tenure', 'monthly_charges']])
    return train, validate, test