# A collection of various tools to help estimate and analyze tail risks.
import pandas as pd



def empirical_expected_shortfall(series, k):
    '''
    Returns the expected shortfall, that is,
    the expectation of X ('series') conditional on X exceeding a threshold 'k'.
    '''
    
    # To get the conditional expectation for X given that X => k, we need to:
    # 1. Integrate P(X)*X from k to inf
    # 2. Normalize by P(X >= k) = integral of P(X) from k to inf; This is the survival function S(k)
    
    # Clean and take absolute values
    cleaned_series = series.dropna().abs()
    
    _part1 = ((cleaned_series >= k) * cleaned_series).mean()
    _part2 = ((cleaned_series >= k)).mean()
    
    return _part1 / _part2



def expected_tail_shortfall(alpha, scale, k):
    '''
    Returns the expected shortfall for a Pareto distribution.
    'alpha' and 'scale' give the tail exponent and scale parameters of the Pareto, respectively.
    'k' gives the threshold to be used.
    To get the conditional expectation for X given that X => k, we need to:
       1. Integrate P(X)*X from k to inf
       2. Normalize by P(X >= k) = integral of P(X) from k to inf; This is the survival function S(k)
    The simple result is k * alpha / (alpha - 1).
    '''
    
    assert alpha > 1, 'Tail exponent alpha has to be larger than 1'
    assert k > scale, 'Threshold has to be larger than scale'
    
    es = k * alpha / (alpha - 1)
    
    return es
