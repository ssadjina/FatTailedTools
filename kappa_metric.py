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

def estimate_alpha_from_kappa_n(kappa, n=1):
    '''
    Estimates tail exponent alpha from the empirical kappa_n.
    Based on Kappa values for Student t distribution from
    "Statistical Consequences of Fat Tails" by Nassim N. Taleb (Table 8.3).
    '''
    
    # Interpolation of kappa values from Table 8.3 in "Statistical Consequences of Fat Tails" by Nassim N. Taleb
    df_kappa = pd.DataFrame({
        'alpha': np.linspace(1.25, 4, 11+1),
        1:   [0.792, 0.647, 0.543, 0.465, 0.406, 0.359, 0.321, 0.29, 0.265, 0.243, 0.225, 0.209],
        30:  [0.765, 0.609, 0.483, 0.387, 0.316, 0.256, 0.224, 0.191, 0.167, 0.149, 0.13, 0.126],
        100: [0.756, 0.587, 0.451, 0.352, 0.282, 0.227, 0.189, 0.159, 0.138, 0.121, 0.1, 0.093],
    }).set_index('alpha').reindex(np.linspace(1.25, 4, 8*11+1)).interpolate()
    
    return np.mean((df_kappa[n] - kappa).abs().sort_values().iloc[:2].index.values)



def estimate_alpha_from_kappa(series, n_bootstrapping=int(1e6)):
    '''
    Estimates tail exponent alpha by estimating the empirical kappa using bootstrapping with "n_bootstrapping" samples.
    '''
    
    assert n_bootstrapping >= 1, 'Must have at least 1 bootstrap sample'
    
    cleaned_series = series.dropna().abs()
    
    # Calculate kappas
    result = []
    for n in [1, 30, 100]:
        
        # Get empirical kappa_n
        _kappa_n = kappa_n(cleaned_series, n=max([2, n]), n_bootstrapping=int(n_bootstrapping))
        
        # Estimate alpha from kappa_n
        _alpha = estimate_alpha_from_kappa_n(_kappa_n, n=n)
        
        result.append(_alpha)
        
    # Return mean
    return np.mean(result)
