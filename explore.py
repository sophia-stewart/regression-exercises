# EXPLORATION FUNCTIONS

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import itertools

def plot_variable_pairs(df, cols, x_vars=[None]):
    '''
    This function takes in a dataframe, a list of two or more columns, and a list/string 
    of any columns to be assigned as the x-variable. It returns lmplots of the columns.
    '''
    for x, y in list(itertools.combinations(cols, 2)):
        if y in x_vars:
            sns.lmplot(data=df, x=y, y=x, line_kws={'color':'red'})
        else:
            sns.lmplot(data=df,x=x, y=y, line_kws={'color':'red'})
    return plt.show();

def months_to_years(df, col):
    '''
    This function takes in a dataframe and a column name; it takes a number of
    months from the inputted column and returns the number of whole years in a new
    column. It returns the original dataframe with the new column added.
    '''
    df[f'{col}_years'] = (df[col] / 12)
    df[f'{col}_years'] = [math.trunc(num) for num in df[f'{col}_years']]
    return df

def plot_categorical_and_continuous_vars(df, cat_cols, cont_cols):
    '''
    This function takes in a dataframe and the names of categorical and continuous
    columns. It returns a stripplot, boxplot, and barplot of the inputted features.
    '''
    for cat_col, cont_col in list(itertools.product(cat_cols, cont_cols)):
        plt.figure(figsize=(12,6))
        plt.subplot(1, 3, 1)
        sns.stripplot(x=cat_col, y=cont_col, data=df)
        plt.subplot(1, 3, 2)
        sns.boxplot(x=cat_col, y=cont_col, data=df)
        plt.subplot(1, 3, 3)
        sns.barplot(x=cat_col, y=cont_col, data=df)
        plt.suptitle(f'{cat_col} and {cont_col}')
        plt.tight_layout()
    return plt.show();