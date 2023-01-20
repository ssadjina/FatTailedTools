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

    # Resample series over periods using the specified offset and forward-fill
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

    # Resample series over periods using the specified offset and forward-fill
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



def get_max_drawdowns(series, periods='1d', offset=0, use_log=True, drop_zeros=True):
    '''
    Returns the maximum drawdowns of a Pandas Series using period "periods" with an offset of 'offset' samples.
    :param series: Pandas Series holding the time series data of prices.
    :param periods: Specifies the time period over which to calculate the maximum drawdowns (e.g. daily, weekly).
    :param offset: Number of samples to shift 'series' by before calculating draws. This can be useful to choose the bias when calculating over time periods longer than the frequency of the price data.
    :param use_log: Whether or not to use logarithmic (or geometric) draws.
    :param drop_zeros: If true, samples where the maximum drawdowns are zero will be dropped before returning. This is useful for avoiding that the results contain a lot of zeros because missing prices were forward-filled before calculating draws.
    :return: A Pandas Series holding the resulting maximum drawdowns over the specified period with specified sample offset.
    '''

    # Drop NAs and apply specified sample offset
    cleaned_series = series.dropna().iloc[offset:]

    # Get the rolling maximum of the series over periods and forward-fill
    rolling_max = cleaned_series.rolling(periods, min_periods=1).max().ffill()

    # Calculate the changes (logarithmic or geometric) from the rolling max
    if use_log:
        drawdowns = np.log(cleaned_series/rolling_max)
    else:
        drawdowns = cleaned_series/rolling_max - 1.0

    # Resample and get the minimum over the periods
    max_drawdowns = drawdowns.resample(periods, label='right').min()

    # Drop zeros introduced by using ffill()
    if drop_zeros:
        max_drawdowns = max_drawdowns.drop(max_drawdowns.index[max_drawdowns == 0])

    return max_drawdowns



def get_max_drawups(series, periods='1d', offset=0, use_log=True, drop_zeros=True):
    '''
    Returns the maximum drawups of a Pandas Series using period "periods" with an offset of 'offset' samples.
    :param series: Pandas Series holding the time series data of prices.
    :param periods: Specifies the time period over which to calculate the maximum drawups (e.g. daily, weekly).
    :param offset: Number of samples to shift 'series' by before calculating draws. This can be useful to choose the bias when calculating over time periods longer than the frequency of the price data.
    :param use_log: Whether or not to use logarithmic (or geometric) draws.
    :param drop_zeros: If true, samples where the maximum drawups are zero will be dropped before returning. This is useful for avoiding that the results contain a lot of zeros because missing prices were forward-filled before calculating draws.
    :return: A Pandas Series holding the resulting maximum drawups over the specified period with specified sample offset.
    '''

    # Drop NAs and apply specified sample offset
    cleaned_series = series.dropna().iloc[offset:]

    # Get the rolling minimum of the series over periods and forward-fill
    rolling_min = cleaned_series.rolling(periods, min_periods=1).min().ffill()

    # Calculate the changes (logarithmic or geometric) from the rolling max
    if use_log:
        drawups = np.log(cleaned_series/rolling_min)
    else:
        drawups = cleaned_series/rolling_min - 1.0

    # Resample and get the maximum over the periods
    max_drawups = drawups.resample(periods, label='right').max()

    # Drop zeros introduced by using ffill()
    if drop_zeros:
        max_drawups = max_drawups.drop(max_drawups.index[max_drawups == 0])

    return max_drawups
