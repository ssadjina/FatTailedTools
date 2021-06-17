# A collection of various tools to help analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from FatTailedTools.plotting import plot_survival_function
from FatTailedTools.survival import get_survival_function

def fit_alpha_linear(series, tail_start_sigma=2, plot=True):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_start_sigma' defines where the tail starts in terms of standard deviations.
    '''
    
    # Get survival function values
    if plot:
        survival, ax = plot_survival_function(series, tail_zoom=True)
    else:
        survival = get_survival_function(series)
    
    # Estimate tail start (= everything beyond 'tail_start_sigma' sigmas)
    tail_start = tail_start_sigma*series.std()
    
    # Get tail
    survival_tail = np.log10(survival.loc[survival['Values'] >= tail_start].iloc[:-1])
    
    # Fit the tail
    tail_fit = np.polyfit(survival_tail['Values'], survival_tail['P'], 1)
    lin_func = np.poly1d(tail_fit)
    
    # Get MAD (mean absolute error)
    mad_error = np.mean(np.abs(np.subtract(lin_func(survival_tail['Values']), survival_tail['Values'])))

    # Plot the fit
    if plot:
        ax.plot(10**survival_tail['Values'], 10**lin_func(survival_tail['Values']), 'r');
        ax.legend(['Fit (MAD = {:.2f})'.format(mad_error), 'Data']);
        plt.title('Tail exponent fitted to tail (alpha = {:.2f})'.format(-tail_fit[0]));
    
    return -tail_fit[0]



from scipy.stats import t, pareto

def fit_alpha(series, plot=True):
    '''
    Estimates the tail parameter by fitting a Pareto or a Studend-T to the data.
    '''
    
    # Is the data only positive?
    if (series.dropna() < 0).sum() == 0:
        # ... then fit a Pareto
        dist = pareto
    else:
        # ... otherwise use Student-T
        dist = t
    
    # Fit the distribution
    params = dist.fit(series.dropna())
    
    if plot:
        _, ax = plot_survival_function(series, distribution=(dist, params));
        plt.title('Tail exponent estimated from fitting (alpha = {:.2f})'.format(-tail_fit[0]));
        
    return params[0]
