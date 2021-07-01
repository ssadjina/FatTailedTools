# A collection of various tools to help analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from FatTailedTools.plotting import plot_survival_function
from FatTailedTools.survival import get_survival_function

def fit_alpha_linear(series, tail_start_mad=3, plot=True):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_start_mad' defines where the tail starts in terms of the mean absolute deviation.
    '''
    
    # Get survival function values
    if plot:
        survival, ax = plot_survival_function(series, tail_zoom=True)
    else:
        survival = get_survival_function(series)
    
    # Estimate tail start (= everything beyond 'tail_start_mad' mean absolute deviations)
    tail_start = tail_start_mad*series.abs().mad()
    
    # Get tail
    survival_tail = np.log10(survival.loc[survival['Values'] >= tail_start].iloc[:-1])
    
    # Fit the tail
    tail_fit = np.polyfit(survival_tail['Values'], survival_tail['P'], 1)
    lin_func = np.poly1d(tail_fit)
    
    # Get MSE (mean squared error)
    mse_error = np.mean(np.square(np.subtract(lin_func(survival_tail['Values']), survival_tail['Values'])))

    # Plot the fit
    if plot:
        ax.plot(10**survival_tail['Values'], 10**lin_func(survival_tail['Values']), 'r');
        ax.legend(['Fit (MSE = {:.2f})'.format(mse_error), 'Data']);
        plt.title('Tail exponent fitted to tail (alpha = {:.2f})'.format(-tail_fit[0]));
    
    return -tail_fit[0]



from scipy.stats import t

def fit_alpha(series, plot=True):
    '''
    Estimates the tail parameter by fitting a Studend-T to the data.
    '''
    
    # Is the data only one-sided?
    if (series.dropna() < 0).sum() * (series.dropna() >= 0).sum() == 0:
        # ... then construct a two-sided distribution
        series = pd.concat([-series.dropna().abs(), series.dropna().abs()])

    # Fit the distribution
    params = t.fit(series.dropna())
    
    if plot:
        _, ax = plot_survival_function(series, distribution=(t, params));
        plt.title('Tail exponent estimated from fitting (alpha = {:.2f})'.format(params[0]));
        
    return params[0]



import seaborn as sns

def fit_alpha_subsampling(series, frac=0.7, n_subsets=100, n_tail_start_samples=1, plot=True):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    Also randomly samples where the tail of the distribution is assumed to start (using 'n_tail_start_samples' samples per subset).
    '''
    
    _alpha_results_both  = []
    _alpha_results_left  = []
    _alpha_results_right = []
    
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        for tail_start_mad in np.random.normal(2.25, 0.45, n_tail_start_samples):
            
            _alpha_results_both.append(subsample.abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False))
            _alpha_results_left.append(subsample.where(subsample  < 0).abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False))
            _alpha_results_right.append(subsample.where(subsample >= 0).abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False))
    
    _alpha_results_both  = pd.Series(_alpha_results_both)
    _alpha_results_left  = pd.Series(_alpha_results_left)
    _alpha_results_right = pd.Series(_alpha_results_right)        
        
    if plot:
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Tail exponents for {} with random subsamples'.format(series.name))
        
        for idx, _alpha_results in enumerate([_alpha_results_both, _alpha_results_left, _alpha_results_right]):
            
            sns.histplot(data=_alpha_results, color=['C7', 'C3', 'C0'][idx], stat='probability', bins=10, ax=ax[idx]);
            ax[idx].set_title('Median = {:.1f} | Mean = {:.1f} ({})'.format(_alpha_results.median(), _alpha_results.mean(), ['both', 'left', 'right'][idx]));
            ax[idx].set_xlabel('Tail exponent ({})'.format(['both', 'left', 'right'][idx]));
            
        plt.show();
    
    alphas = pd.concat([_alpha_results_both, _alpha_results_left, _alpha_results_right], axis=1, keys=['Both', 'Left', 'Right'])
    
    return alphas
