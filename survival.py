# Methods to calculate survival functions and survival probabilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings



from FatTailedTools import alpha



def get_survival_function(series, inclusive=True):
    '''
    Calculates a (one-sided) survival function from (the absolute values of) a Pandas Series 'series'.
    Returns a Pandas DataFrame with the columns "Values", X, and "P", P(x >= X), keeping the index (and NAs dropped).
    If 'inclusive', P(x >= X) is used and the largest data point is plotted. Else, P(x > X) is used and the largest data point is not plotted. The latter is consistent with, e.g., seaborn.ecdfplot(complementary=True).
    '''
    
    # Take absolute values and drop NAs
    abs_series = series.dropna().abs()
    
    # Set up DataFrame and sort values from largest to smallest
    survival = pd.DataFrame(abs_series.values, index=abs_series.index, columns=['Values']).sort_values(by='Values', ascending=True)
    
    # Determine whether we compare with '>=' or with '>'
    if inclusive:
        func = lambda x: (survival['Values'] >= x).mean()
    else:
        func = lambda x: (survival['Values'] > x).mean()
    
    # Get survival probabilities
    survival['P'] = survival['Values'].apply(func)
    
    return survival



def get_tail_survival_probability(series, X, tail_frac=None, plot=True):
    '''
    Returns the empirical survival probability for x >= 'X'.
    Uses a linear fit of the tail of the log-log survival function of the distribution given by 'series'.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    If 'plot' then the linear fit and the calculated probability are visualized.
    '''
    
    # Get start of tail and check that X is in tail
    tail_start = alpha.get_tail_start(series, tail_frac)
    
    # Fit tail
    alpha_fit, scale = alpha.fit_alpha_linear(series, return_scale=True, tail_frac=tail_frac, plot=plot)
    
    # Probability that x >= X
            
    # If X is not in tail (smaller)...
    if X < tail_start:
        
        # Get survival function...
        survival = get_survival_function(series, inclusive=True)
        
        # ...and estimate P as smallest P for any x smaller than X
        probability = survival.loc[survival['Values'] < X, 'P'].min()
        
    # Otherwise estimate from tail fit
    else:
        probability = get_tail_survival_probability_from_alpha_fit(X, alpha_fit, scale, tail_start=tail_start)
    
    # Extend plot of the fitted function
    if plot:
        plt.vlines(X, 1e-16, 1e1, 'k', linestyles='dashed', alpha=0.4)
        if probability is not np.nan:
            plt.hlines(probability, 1e-16, 1e1, 'k', linestyles='dashed', alpha=0.4)
    
    return probability



def get_tail_survival_probability_from_alpha_fit(X, alpha_fit, scale, tail_start):
    '''
    Returns the probability that x >= 'X' given the tail exponent 'alpha_fit' and the scale 'scale'.
    'tail_start' must be passed to make sure that 'X' actually is in the tail.
    
    '''
    
    # Check that 'x' is in tail
    if X < tail_start:
        warnings.warn('X={} is not in the tail (which is estimated to start at {}). Returning \'np.nan\''.format(X, tail_start))
        return np.nan
    
    probability = 10**(alpha_fit * (np.log10(scale/X)))
    
    return np.clip(probability, a_min=0, a_max=1)



import seaborn as sns
from matplotlib.ticker import PercentFormatter

def get_survival_probability_subsampling(series, X, frac=0.7, n_subsets=300, tail_frac_range=None, plot=True, title_annotation=None):
    '''
    Estimates the empirical survival probability for x >= 'X' using subsampling.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction of samples kept.
    'tail_frac_range' defines what uniform range to draw from as a guess for where the tail starts
    in terms of the fraction of data used (from largest to smallest).
    Depending on the location of 'X', either uses a linear fit of the tail of the log-log survival function 
    of the distribution given by 'series', or a naive estimate using the empirical survival probabilities.
    If 'plot' the shows histogram of the results with a 'title_annotation' if given.
    '''
    
    # Prepare array to save results
    results = []
    
    # When no tail_frac_range is given a simple heuristic is used:
    if tail_frac_range is None:
        tail_frac_midle = alpha.get_tail_frac_guess(series)
        tail_frac_range = (tail_frac_midle/1.5, tail_frac_midle*1.5)

    # Subsample
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        # Randomly choose a tail start from a uniform random distribution
        tail_frac = np.random.uniform(*tail_frac_range)
            
        # Get estimate for where tail starts
        tail_start = alpha.get_tail_start(subsample, tail_frac)
        
        # If X is not in tail (smaller)...
        if X < tail_start:
            
            # Get survival function...
            survival = get_survival_function(subsample, inclusive=True)
            
            # ...and estimate P as smallest P for any x smaller than X
            result = survival.loc[survival['Values'] < X, 'P'].min()
            
        # Otherwise estimate with tail fit
        else:
            result = get_tail_survival_probability(subsample, X, tail_frac=tail_frac, plot=False)
            
        plt.show();
        results.append(result)
        
    results = pd.Series(results)
    
    # Plot
    if plot:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        results.hist(bins=10);
        plt.xlabel('Survival probability');
        plt.title('Survival probability P(|x| >= {:.3f}) ({})'.format(X, series.name));
        plt.vlines(x=results.mean(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', label='Mean ({:.2%})'.format(results.mean()));
        plt.vlines(x=results.median(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', linestyle='--', label='Median ({:.2%})'.format(results.median()));
        plt.legend();
        
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    
    
    return results
