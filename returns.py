# A collection of tools useful to calculate things such as returns and drawdowns for financial time-series data.
import pandas as pd
import numpy as np



def get_returns(series, periods='1d', offset=0, drop_zeros=True):
    '''
    Returns the geometric returns from a given Pandas Series using period "periods" with an offset of 'offset' samples.
    :param series: Pandas Series holding the time series data of prices.
    :param periods: Specifies the time period over which to calculate returns (e.g. daily, weekly).
    :param offset: Number of samples to shift 'series' by before calculating returns. This can be useful to choose the
    bias when calculating returns over time periods longer than the frequency of the price data.
    :param drop_zeros: If true, samples where the returns are zero will be dropped before returning. This is useful
    for avoiding that the results contain a lot of zeros because missing prices were forward-filled before
    calculating returns.
    :return: A Pandas Series holding the resulting returns over the specified period with specified sample offset.
    '''

    # Drop NAs
    cleaned_series = series.dropna()

    # Resample series over periods using the specified offset and ffill
    series_resampled = cleaned_series.iloc[offset:].resample(periods).first().ffill()

    # Calculate returns
    returns = series_resampled/series_resampled.shift(1) - 1

    # Drop zeros introduced by using ffill()
    if drop_zeros:
        returns = returns.drop(returns.index[returns == 0])

    # Make sure the product of all returns is equal the total return over the period
    series_reconstructed = cleaned_series.iloc[offset] * (1 + returns).cumprod()
    rel_error = (series_reconstructed / series - 1).abs().sum()
    assert rel_error <= 1e-8, 'Warning! Sum of log returns was not equal to log return over entire period (relative error: {})'.format(rel_error)

    return returns



def get_log_returns(series, periods='1d', offset=0, drop_zeros=True):
    '''
    Returns the logarithmic returns from a given Pandas Series using period "periods" with an offset of 'offset' samples.
    :param series: Pandas Series holding the time series data of prices.
    :param periods: Specifies the time period over which to calculate log returns (e.g. daily, weekly).
    :param offset: Number of samples to shift 'series' by before calculating log returns. This can be useful to choose the bias when calculating log returns over time periods longer than the frequency of the price data.
    :param drop_zeros: If true, samples where the log returns are zero will be dropped before returning. This is useful for avoiding that the results contain a lot of zeros because missing prices were forward-filled before calculating log returns.
    :return: A Pandas Series holding the resulting log returns over the specified period with specified sample offset.
    '''

    # Drop NAs
    cleaned_series = series.dropna()

    # Resample series over periods using the specified offset and ffill
    series_resampled = cleaned_series.iloc[offset:].resample(periods).first().ffill()

    # Calculate log returns
    log_returns = np.log(series_resampled/series_resampled.shift(1))

    # Drop zeros introduced by using ffill()
    if drop_zeros:
        log_returns = log_returns.drop(log_returns.index[log_returns == 0])

    # Make sure the sum of all log returns is equal the total log return over the period
    series_reconstructed = cleaned_series.iloc[offset] * np.exp(log_returns.cumsum())
    rel_error = (series_reconstructed / series - 1).abs().sum()
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
