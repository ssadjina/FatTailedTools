# A collection of tools useful to calculate things such as returns and drawdowns for financial time-series data.
import pandas as pd
import numpy as np



def get_returns(series, periods='1d'):
    '''
    Calculates the returns from a given Pandas Series using period "periods"
    '''
    
    # Resample series over periods
    series_resampled = series.resample(periods).last()
    
    # Calculate returns
    returns = series_resampled/series_resampled.shift(1) - 1
    
    # Make sure the product of all returns is equal the total return over the period
    rel_error = returns.product() * series_resampled.dropna().iloc[0] / series_resampled.dropna().iloc[-1] - 1
    assert rel_error <= 1e-8, 'Warning! Product of returns was not equal to return over entire period (relative error: {})'.format(rel_error)
    
    return returns



def get_log_returns(series, periods='1d'):
    '''
    Returns the logarithmic return from a given Pandas Series using period "periods"
    '''
    
    # Resample series over periods
    series_resampled = series.resample(periods).last()
    
    # Calculate log returns
    log_returns = np.log(series_resampled/series_resampled.shift(1))
    
    # Make sure the sum of all log returns is equal the total log return over the period
    rel_error = np.exp(log_returns.sum()) * series_resampled.dropna().iloc[0] / series_resampled.dropna().iloc[-1] - 1
    assert rel_error <= 1e-8, 'Warning! Sum of log returns was not equal to log return over entire period (relative error: {})'.format(rel_error)
    
    return log_returns



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
