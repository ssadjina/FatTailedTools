# A collection of various tools to help estimate and analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from FatTailedTools import survival
from FatTailedTools import plotting



def get_tail_start(series, tail_frac=0.1):
    '''
    Returns the start of the tail of 'series' based on 'tail_frac'.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    '''
    
    cleaned_series = series.dropna().abs()
    
    return cleaned_series.sort_values(ascending=False).iloc[:int(np.ceil(len(cleaned_series)*tail_frac))].iloc[-1]



def fit_alpha_linear(series, tail_frac=0.1, plot=True, return_loc=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    The estimated location of the Pareto (with the estimated tail exponent) will also re returned if 'return_loc' is True.
    '''
    
    # Get survival function values
    if plot:
        survival_func, ax = plotting.plot_survival_function(series, tail_zoom=False)
    else:
        survival_func = survival.get_survival_function(series)
    
    # Estimate tail start (= everything beyond 'tail_start_mad' mean absolute deviations)
    tail_start = get_tail_start(series, tail_frac=tail_frac)
    
    # Get tail
    survival_tail = np.log10(survival_func.loc[survival_func['Values'] >= tail_start].iloc[:-1])
    
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
        x_values = np.log10(np.array([survival_func.loc[survival_func['Values'] > 0, 'Values'].min()/10, survival_func.loc[survival_func['Values'] > 0, 'Values'].max()*10]))
        
        # Visualize area where tail is assumed to be
        ax.axvspan(10**survival_tail['Values'].min(), 10**(survival_func['Values'].max() + 1), survival_func['P'].min()/10, 1, alpha=0.1, color='r')
        
        # Plot tail fit
        ax.plot(10**x_values, 10**lin_func(x_values), 'r--', alpha=0.6);
        
        # Legend and title
        ax.legend(['Data', 'Tail', 'Fit (MSE = {:.2f})'.format(mse_error)]);
        plt.title('Tail exponent for {} fitted to tail (alpha = {:.2f}, loc = {:.2f})'.format(series.name, tail, location));
        
    # Construct result
    result = tail, location if return_loc else tail
    
    return result



from scipy.stats import t

def fit_alpha(series, plot=True, return_additional_params=False, **kwargs):
    '''
    Estimates the tail parameter by fitting a Studend-T to the data.
    If the passed data is from a one-sided distribution, it will first be mirrored at 0 to make it symmetrical.
    '''
    
    # Is the data only one-sided?
    if (series.dropna() < 0).sum() * (series.dropna() > 0).sum() == 0:
        # ... then construct a two-sided distribution
        series = pd.concat([-series.dropna().abs(), series.dropna().abs()])

    # Fit the distribution
    params = t.fit(series.dropna(), **kwargs)
    
    if plot:
        _, ax = plotting.plot_survival_function(series, distribution=(t, params));
        plt.title('Tail exponent estimated from fitting (alpha = {:.2f})'.format(params[0]));
        
    return params[0] if not return_additional_params else params



import seaborn as sns

def fit_alpha_linear_subsampling(series, frac=0.7, n_subsets=300, tail_frac=0.1, plot=True, return_loc=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    If return_loc is True, also returns where the tail of the distribution is assumed to start.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    '''
    
    # Set up lists
    _results_both  = []
    _results_left  = []
    _results_right = []
    
    # Subsample and fit
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        _results_both.append(subsample.abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))
        _results_left.append(subsample.where(subsample  < 0).abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))
        _results_right.append(subsample.where(subsample >= 0).abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))      
        
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



def max_likelihood_pareto_alpha(series, min_samples=5, plot=False):
    '''
    Estimates the maximum likelihood for the tail exponent of a Pareto distribution.
    If 'plot' visualizes the convergence of the estimator with increasing number of samples.
    'min_samples' determines the number of samples required. 
    '''
    
    # Drop NAs and take absolute values
    cleaned_series = abs(series.dropna())
    
    # Make sure we have enough data (relative to min_samples)
    assert len(cleaned_series) >= min_samples, 'Too few samples (n = {}). Get more data or decrease \'min_samples\'.'.format(len(cleaned_series))
    
    if plot:
    
        # Apply maximum likelihood estimation for alpha (assuming a Pareto distribution)
        # Do so using an expanding window starting with the largest values
        estimation = cleaned_series.sort_values(ascending=False).expanding(min_periods=min_samples).apply(
            lambda x: 1 / (np.log(x).mean() - np.log(x.min()))
        )

        estimation.reset_index(drop=True).plot(grid=True, linestyle='--');
        plt.xlabel('Number of samples (descending order)');
        plt.ylabel('alpha');
        plt.title('Maximum Likelihood Estimation of Tail Exponent');
        plt.hlines(y=estimation.iloc[-1], xmin=0, xmax=len(cleaned_series));
        plt.legend(['Estimator', 'Result ({:.2f})'.format(estimation.iloc[-1])]);
        
        return estimation.iloc[-1]
        
    else:
    
        return 1 / (np.log(cleaned_series).mean() - np.log(cleaned_series.min()))


    
def max_likelihood_alpha(series, min_samples=5, plot=False, tail_frac=0.1):
    '''
    Estimates the maximum likelihood for the tail exponent.
    If 'plot' visualizes the convergence of the estimator with increasing number of samples.
    'min_samples' determines the number of samples required. 
    'tail_frac' gives the fraction of data used (from largest to smallest) that is assumed to constitute the power law tail.
    '''
    
    # Drop NAs and take absolute values
    cleaned_series = abs(series.dropna())
    
    # Make sure we have enough data (relative to min_samples)
    assert len(cleaned_series) >= np.ceil(min_samples/tail_frac), 'Too few samples (n = {}). Get more data or decrease \'min_samples\'.'.format(len(cleaned_series))   
    
    # Take the largest 'tail_frac' samples (assumed to constitute the tail) and estimate alpha from maximum likelihood
    result = max_likelihood_pareto_alpha(
        series=cleaned_series.sort_values(ascending=False).iloc[:int(len(cleaned_series)*tail_frac)],
        min_samples=min_samples,
        plot=plot
    )
    
    return result



def max_likelihood_alpha_subsampling(series, frac=0.7, n_subsets=300, tail_frac=0.1, plot=True):
    '''
    Estimates the tail parameter via maximum likelihood for alpha assuming a power law tail.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    '''
    
    # Set up lists
    _results = []
    
    # Subsample and fit
    for subsample in [series.dropna().sample(frac=frac) for i in range(n_subsets)]:
        
        _results.append(subsample.agg(max_likelihood_alpha, tail_frac=tail_frac, plot=False))
        
    # Assemble into DataFrame
    results = pd.Series(_results)
        
    # Plot
    if plot:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        results.hist(bins=10);
        plt.xlabel('alpha');
        plt.title('Maximum Likelihood Estimation of Tail Exponent for {} with random subsamples'.format(series.name));
        plt.vlines(x=results.mean(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', label='Mean ({:.2f})'.format(results.mean()));
        plt.vlines(x=results.median(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', linestyle='--', label='Median ({:.2f})'.format(results.median()));
        plt.legend();
    
    return results
