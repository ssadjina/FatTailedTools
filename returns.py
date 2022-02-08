# A collection of tools useful to calculate things such as returns and drawdowns for financial time-series data.
import pandas as pd
import numpy as np



def get_returns(series, periods=1):
    '''
    Calculates the returns from a given Pandas Series using period "periods"
    '''
    return series.pct_change(periods=periods, fill_method=None).dropna()



def get_log_returns(series, periods=1):
    '''
    Returns the logarithmic return from a given Pandas Series using period "periods"
    '''
    return np.log(series/series.shift(periods=periods)).dropna()



def get_returns_from_log_returns(series):
    '''
    Converts log returns to returns
    '''
    return np.exp(series) - 1



def get_log_returns_from_returns(series):
    '''
    Converts returns to log returns
    '''
    return np.log(series + 1)



def get_max_drawdowns(series, n, use_log=True):
    '''
    Returns the maximum drawdowns of a Pandas Series using "n" periods.
    If "use_log", uses logarithmic returns/drawdowns.
    '''
    
    cleaned_series = series.dropna()
    
    # Calculate the max drawdown in the past window days for each day in the series.
    Roll_Max = cleaned_series.rolling(str(n) + 'd', min_periods=1).max()
    
    if use_log:
        Daily_Drawdown = np.log(cleaned_series/Roll_Max)
    else:
        Daily_Drawdown = cleaned_series/Roll_Max - 1.0
    
    # Next we calculate the minimum (negative) daily drawdown in that window.
    return Daily_Drawdown.resample(str(n) + 'd').min()



def get_max_drawups(series, n, use_log=True):
    '''
    Returns the maximum drawups of a Pandas Series using "n" periods.
    If "use_log", uses logarithmic returns/drawups.
    '''
    
    cleaned_series = series.dropna()
    
    # Calculate the max drawup in the past window days for each day in the series.
    Roll_Min = cleaned_series.rolling(str(n) + 'd', min_periods=1).min()
    
    if use_log:
        Daily_Drawup = np.log(cleaned_series/Roll_Min)
    else:
        Daily_Drawup = cleaned_series/Roll_Min - 1.0
    
    # Next we calculate the maximum (positive) daily drawup in that window.
    return Daily_Drawup.resample(str(n) + 'd').max()
