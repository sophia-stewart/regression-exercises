# ZILLOW WRANGLING FUNCTIONS

import pandas as pd

def acquire_zillow():
    '''
    This function takes no arguments and returns a dataframe of 2017 Single Family 
    Residential property data from Zillow. It searches for a csv file (zillow.csv)
    with the requested data and reads that file into a dataframe. If the csv
    file is not found, it retrieves the SQL query result (using data from env.py)
    and reads it into a dataframe. It then caches this data into a csv file.
    '''
    import os
    if os.path.isfile('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col=0)
        return zillow
    else:
        from env import user, password, host
        url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
        sql = '''
        SELECT bedroomcnt, 
               bathroomcnt,
               calculatedfinishedsquarefeet,
               taxvaluedollarcnt,
               yearbuilt,
               taxamount,
               fips
        FROM properties_2017
        WHERE propertylandusetypeid IN (261, 279);
        '''
        zillow = pd.read_sql(sql, url)
        zillow.to_csv('zillow.csv')
        return zillow

def remove_outliers(df, cols, k):
    '''
    This function takes in a dataframe, a list of columns from that dataframe,
    and a k-value (int) which is used to specify the upper and lower bounds 
    for removing outliers. It returns the dataframe with the outliers removed.
    '''
    for col in cols:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        upper = q3 + k * iqr
        lower = q1 - k * iqr
        df = df[(df[col] > lower) & (df[col] < upper)]
    return df

def prep_zillow(zillow):
    zillow = zillow.dropna()
    zillow.fips = zillow['fips'].astype('O') 
    zillow.yearbuilt = zillow['yearbuilt'].astype('O')
    zillow.rename(columns={'bedroomcnt':'beds', 
                   'bathroomcnt':'baths', 
                   'calculatedfinishedsquarefeet':'sq_ft', 
                   'taxvaluedollarcnt':'tax_value'}, inplace=True)
    cols = zillow.select_dtypes(float)
    zillow = remove_outliers(zillow, cols, 1.5)
    return zillow

def wrangle_zillow():
    prep_zillow(acquire_zillow())
    return zillow