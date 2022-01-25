# A collection of various tools to help estimate and analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_tail_start(series, tail_start_mad):
    '''
    Returns the start of the tail of 'series' based on 'tail_start_mad'.
    'tail_start_mad' defines where the tail starts in terms of the mean absolute deviation (typically between 2-4 MADs).
    '''
    
    return tail_start_mad * series.abs().mad()



from FatTailedTools.survival import get_survival_function
from FatTailedTools.plotting import plot_survival_function



def fit_alpha_linear(series, tail_start_mad=2.5, plot=True, return_loc=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_start_mad' defines where the tail starts in terms of the mean absolute deviation (typically between 2-4 MADs).
    The estimated location of the Pareto (with the estimated tail exponent) will also re returned if 'return_loc' is True.
    '''
    
    # Get survival function values
    if plot:
        survival, ax = plot_survival_function(series, tail_zoom=False)
    else:
        survival = get_survival_function(series)
    
    # Estimate tail start (= everything beyond 'tail_start_mad' mean absolute deviations)
    tail_start = get_tail_start(series, tail_start_mad)
    
    # Get tail
    survival_tail = np.log10(survival.loc[survival['Values'] >= tail_start].iloc[:-1])
    
    # Fit the tail
    tail_fit = np.polyfit(survival_tail['Values'], survival_tail['P'], 1)
    lin_func = np.poly1d(tail_fit)
    
    # Get tail parameter and location/scale
    tail = -tail_fit[0]
    location = (1 - tail_fit[1]) / tail_fit[0]
    
    # Get MSE (mean squared error)
    mse_error = np.mean(np.square(np.subtract(lin_func(survival_tail['Values']), survival_tail['Values'])))

    # Plot the fit
    if plot:
        
        # Get x values (from min/max of survival values)
        x_values = np.log10(np.array([survival.loc[survival['Values'] > 0, 'Values'].min()/10, survival.loc[survival['Values'] > 0, 'Values'].max()*10]))
        
        # Visualize area where tail is assumed to be
        ax.axvspan(10**survival_tail['Values'].min(), 10**(survival['Values'].max() + 1), survival['P'].min()/10, 1, alpha=0.1, color='r')
        
        # Plot tail fit
        ax.plot(10**x_values, 10**lin_func(x_values), 'r--', alpha=0.6);
        
        # Legend and title
        ax.legend(['Data', 'Tail', 'Fit (MSE = {:.2f})'.format(mse_error)]);
        plt.title('Tail exponent fitted to tail (alpha = {:.2f}, loc = {:.2f})'.format(tail, location));
        
    # Construct result
    result = tail, location if return_loc else tail
    
    return result



from scipy.stats import t

def fit_alpha(series, plot=True):
    '''
    Estimates the tail parameter by fitting a Studend-T to the data.
    If the passed data is from a one-sided distribution, it will first be mirrored at 0 to make it symmetrical.
    '''
    
    # Is the data only one-sided?
    if (series.dropna() < 0).sum() * (series.dropna() > 0).sum() == 0:
        # ... then construct a two-sided distribution
        series = pd.concat([-series.dropna().abs(), series.dropna().abs()])

    # Fit the distribution
    params = t.fit(series.dropna())
    
    if plot:
        _, ax = plot_survival_function(series, distribution=(t, params));
        plt.title('Tail exponent estimated from fitting (alpha = {:.2f})'.format(params[0]));
        
    return params[0]



import seaborn as sns

def fit_alpha_subsampling(series, frac=0.7, n_subsets=300, n_tail_start_samples=1, plot=True, return_loc=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    If return_loc is True, also returns where the tail of the distribution is assumed to start (using random subsampling with 'n_tail_start_samples' samples per subset).
    '''
    
    # Set up lists
    _results_both  = []
    _results_left  = []
    _results_right = []
    
    # Subsample and fit
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        for tail_start_mad in np.random.normal(2.5, 0.5, n_tail_start_samples):
            
            _results_both.append(subsample.abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False, return_loc=True))
            _results_left.append(subsample.where(subsample  < 0).abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False, return_loc=True))
            _results_right.append(subsample.where(subsample >= 0).abs().agg(fit_alpha_linear, tail_start_mad=tail_start_mad, plot=False, return_loc=True))      
        
    # Assemble into DataFrame
    alphas = pd.DataFrame.from_records(np.hstack([_results_both, _results_left, _results_right]), columns=pd.MultiIndex.from_product([['Both', 'Left', 'Right'], ['Tail Exponent', 'Location']]))    
        
    # Plot
    if plot:
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        fig.suptitle('Tail exponents for {} with random subsamples'.format(series.name))
        
        for idx, name in enumerate(['Both', 'Left', 'Right']):
            
            sns.histplot(data=alphas[(name, 'Tail Exponent')], color=['C7', 'C3', 'C0'][idx], stat='probability', bins=10, ax=ax[idx]);
            ax[idx].set_title('Median = {:.1f} | Mean = {:.1f} ({})'.format(alphas[(name, 'Tail Exponent')].median(), alphas[(name, 'Tail Exponent')].mean(), ['both', 'left', 'right'][idx]));
            ax[idx].set_xlabel('Tail exponent ({})'.format(['both', 'left', 'right'][idx]));
            
        plt.show();
        
        # Also plot locations if return_loc
        if return_loc:
        
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
            fig.suptitle('Locations for {} with random subsamples'.format(series.name))
            
            for idx, name in enumerate(['Both', 'Left', 'Right']):
                
                sns.histplot(data=alphas[(name, 'Location')], color=['C7', 'C3', 'C0'][idx], stat='probability', bins=10, ax=ax[idx]);
                ax[idx].set_title('Median = {:.1f} | Mean = {:.1f} ({})'.format(alphas[(name, 'Location')].median(), alphas[(name, 'Location')].mean(), ['both', 'left', 'right'][idx]));
                ax[idx].set_xlabel('Location ({})'.format(['both', 'left', 'right'][idx]));
                
            plt.show();
        
    # Construct result
    result = alphas if return_loc else alphas.loc[:, (slice(None), 'Tail Exponent')]
    
    return result
