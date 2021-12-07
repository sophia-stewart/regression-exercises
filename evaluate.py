# FUNCTIONS FOR EVALUATION

from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error

def plot_residuals(df, y, yhat):
    '''
    This function takes in a dataframe and the names of a target column and a predictions
    column. It returns a scatterplot of the model's residuals against the target.
    '''
    df['residual'] = df[y] - df[yhat]
    plt.scatter(df[y], df.residual)
    plt.axhline(y=0, ls=':')
    plt.ylabel('residual')
    plt.xlabel(f'{y}')
    plt.title('Model Residuals')
    return plt.show();

def regression_errors(df, y, yhat):
    '''
    This function takes in a dataframe and the names of a target column and a predictions
    column. It returns the model's SSE, ESS, TSS, MSE, and RMSE, in that order.
    '''
    df['residual^2'] = df.residual**2
    SSE = df['residual^2'].sum()
    ESS = sum((df[yhat] - df[y].mean())**2)
    TSS = ESS + SSE
    MSE = SSE/len(df)
    RMSE = math.sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(df, y):
    '''
    This function takes in a dataframe and the name of a target column. It returns the
    SSE, MSE, and RMSE for the baseline, in that order.
    '''
    df['baseline'] = df[y].mean()
    bMSE = mean_squared_error(df[y], df.baseline)
    bSSE = bMSE * len(df)
    bRMSE = math.sqrt(bMSE)
    return bSSE, bMSE, bRMSE

def better_than_baseline(df, y, yhat):
    '''
    This function takes in a dataframe and the names of a target column and a predictions
    column. It returns whether or not the created model is better than the baseline.
    '''
    SSE, ESS, TSS, MSE, RMSE = regression_errors(df, y, yhat)
    bSSE, bMSE, bRMSE = baseline_mean_errors(df, y)
    if bSSE < SSE:
        return False
    else:
        return True