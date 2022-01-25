# Methods to calculate survival functions and survival probabilities
import pandas as pd
import numpy as np
import warnings



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



from FatTailedTools.alpha import get_tail_start, fit_alpha_linear

def get_tail_survival_probability(series, X, tail_start_mad=2.5, plot=True):
    '''
    Returns the empirical survival probability for x >= 'X'.
    Uses a linear fit of the tail of the log-log survival function of the distribution given by 'series'.
    'tail_start_mad' gives the location of the start of the tail in mean absolute deviations.
    If 'plot' then the linear fit and the calculated probability are visualized.
    '''
    
    # Get start of tail and check that X is in tail
    tail_start = get_tail_start(series, tail_start_mad)
    
    # Fit tail
    alpha, loc = fit_alpha_linear(series, return_loc=True, tail_start_mad=tail_start_mad, plot=plot)
    
    # Probability that x >= X
    probability = get_tail_survival_probability_from_alpha_fit(X, alpha, loc, tail_start=tail_start)
    
    # Extend plot of the fitted function
    if plot:
        plt.vlines(X, 1e-16, 1e1, 'k', linestyles='dashed', alpha=0.4)
        if probability is not np.nan:
            plt.hlines(probability, 1e-16, 1e1, 'k', linestyles='dashed', alpha=0.4)
    
    return probability



def get_tail_survival_probability_from_alpha_fit(X, alpha, loc, tail_start):
    '''
    Returns the probability that x >= 'X' given the tail exponent 'alpha' and the location 'loc'.
    'tail_start' must be passed to make sure that 'X' actually is in the tail.
    
    '''
    
    # Check that 'x' is in tail
    if X < tail_start:
        warnings.warn('X={} is not in the tail (which is estimated to start at {}). Returning \'np.nan\''.format(x, tail_start))
        return np.nan
    
    probability = 10**(-alpha * np.log10(X) + (1 + loc * alpha))
    
    return np.clip(probability, a_min=0, a_max=1)



import seaborn as sns
from matplotlib.ticker import PercentFormatter

from FatTailedTools.plotting import plot_survival_function

def get_survival_probability_subsampling(series, X, frac=0.7, n_subsets=300, n_tail_start_samples=1, plot=True, title_annotation=None):
    '''
    Estimates the empirical survival probability for x >= 'X' using subsampling.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    Also uses random subsampling with 'n_tail_start_samples' samples per subset to vary the start of the tail.
    Depending on the location of 'X', either uses a linear fit of the tail of the log-log survival function 
    of the distribution given by 'series', or a naive estimate using the empirical survival probabilities.
    If 'plot' the shows histogram of the results with a 'title_annotation' if given.
    '''
    
    # Prepare array to save results
    results = []

    # Subsample
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        # Choose tail_start_mad
        for tail_start_mad in np.random.normal(2.5, 0.5, n_tail_start_samples):
            
            # Get estimate for where tail starts
            tail_start = get_tail_start(subsample, tail_start_mad)
            
            # If X is not in tail (smaller)...
            if X < tail_start:
                
                # Get survival function...
                survival = get_survival_function(subsample, inclusive=True)
                
                # ...and estimate P as smallest P for any x smaller than X
                result = survival.loc[survival['Values'] < X, 'P'].min()
                
            # Otherwise estimate with tail fit
            else:
                result = get_tail_survival_probability_from_alpha_fit(subsample, X, tail_start_mad=tail_start_mad, plot=False)
                
            plt.show();
            results.append(result)
        
    results = pd.Series(results)
    
    if plot:
        ax = sns.histplot(data=results, stat='probability', bins=10);

        series_name = series.name
        if title_annotation is not None:
            series_name += ' {}'.format(title_annotation)
        
        ax.set_title('Survival probability P(|x| >= {:.3f})\n{}\nMedian = {:.2%} | IQR = {:.2%}'.format(X, series_name, results.median(), results.describe()['75%'] - results.describe()['25%']));
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    
        plt.show();
    
    return results
