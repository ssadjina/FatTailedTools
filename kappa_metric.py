# The kappa metric as introduced by Nassim Taleb in "Statistical Consequences of Fat Tails".
import numpy as np
import pandas as pd



def get_bootstrapped_samples(series, n, n_bootstrapping=10000):
    '''
    Creates a Pandas DataFrame of length n sampled by bootstrapping with "n_bootstrapping" samples/times from data given as a Pandas Series "series".
    '''
    
    return pd.DataFrame(np.random.choice(series, size=(n_bootstrapping, n), replace=True))



def empirical_kappa(series, n0, n, n_bootstrapping=10000):
    '''
    Estimate the empirical kappa using bootstrapping with "n_bootstrapping" samples.
    '''
    
    assert n > n0, 'n must be larger than n0'
    assert n0 >= 1, 'n0 must be >= 1'
    assert n_bootstrapping >= 1, 'Must have at least 1 bootstrap sample'
    
    cleaned_series = series.dropna()
    
    # Create bootstrapped samples
    # Expectation Operator is the sum over all finite outcomes x_i of a random variable X weighted by the probability of occurance p_i
    # To estimate the expectation value, we can use bootstrapping: Draw n_bootstrapping times from the data with replacement
    S_n  = get_bootstrapped_samples(cleaned_series, n, n_bootstrapping).sum(axis=1)
    S_n0 = get_bootstrapped_samples(cleaned_series, n0, n_bootstrapping).sum(axis=1)
    
    # Absolute mean deviation from the mean
    M_n  = (S_n  - S_n.mean() ).abs().mean()
    M_n0 = (S_n0 - S_n0.mean()).abs().mean()
    
    # Calculate kappa
    kappa = 2 - (np.log(n) - np.log(n0)) / np.log(M_n / M_n0)
    
    return kappa



def kappa_n0(series, n0, n_bootstrapping=10000):
    '''
    Estimate the empirical kappa_n_0 using bootstrapping with "n_bootstrapping" samples.
    '''
    
    return empirical_kappa(series, n0, n=n0+1, n_bootstrapping=n_bootstrapping)



def kappa_n(series, n, n_bootstrapping=10000):
    '''
    Estimate the empirical kappa_n using bootstrapping with "n_bootstrapping" samples.
    '''
    
    return empirical_kappa(series, 1, n, n_bootstrapping=n_bootstrapping)
