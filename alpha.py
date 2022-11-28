# A collection of various tools to help estimate and analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from FatTailedTools import survival
from FatTailedTools import plotting



def get_tail_start(series, tail_frac=None):
    '''
    Returns the start of the tail of 'series' based on 'tail_frac'.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    '''
    
    cleaned_series = series.dropna().abs()
    
    # When no tail_frac is given a simple heuristic is used:
    if tail_frac is None:
        tail_frac = get_tail_frac_guess(series)
    
    return cleaned_series.sort_values(ascending=False).iloc[:int(np.ceil(len(cleaned_series)*tail_frac))].iloc[-1]



def fit_alpha_linear(series, tail_frac=None, plot=True, return_scale=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
    The estimated scale of the Pareto (with the estimated tail exponent) will also re returned if 'return_scale' is True.
    '''
    
    # Get survival function values
    if plot:
        survival_func, ax = plotting.plot_survival_function(series, tail_zoom=False)
    else:
        survival_func = survival.get_survival_function(series)
    
    # When no tail_frac is given a simple heuristic is used:
    if tail_frac is None:
        tail_frac = get_tail_frac_guess(series)
    
    # Estimate tail start (= everything beyond 'tail_start_mad' mean absolute deviations)
    tail_start = get_tail_start(series, tail_frac=tail_frac)
    
    # Get tail
    survival_tail = np.log10(survival_func.loc[survival_func['Values'] >= tail_start].iloc[:-1])
    
    # Fit the tail
    tail_fit = np.polyfit(survival_tail['Values'], survival_tail['P'], 1)
    lin_func = np.poly1d(tail_fit)
    
    # Get tail parameter and scale of the Pareto distribution
    tail = -tail_fit[0]
    scale = 10**(- tail_fit[1] / tail_fit[0])
    
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
        plt.title('Tail exponent for {} fitted to tail (alpha = {:.2f}, scale = {:.4f})'.format(series.name, tail, scale));
        
    # Construct result
    result = tail, scale if return_scale else tail
    
    return result



def fit_alpha_linear_fast(series):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log survival function.
    Optimized for speed to be used with, for example, subsampling and Monte Carlo simulations.
    '''
    
    # Get survival function
    abs_series = series.dropna().abs().sort_values(ascending=True).values
    x = np.log10(abs_series)
    y = np.log10(
        [(abs_series >= value).mean() for value in abs_series]
    )

    # Calculate slope
    n = len(x)
    xy = x * y
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = xy.sum()
    sum_x2 = (x**2).sum()
    slope = n * sum_xy - sum_x * sum_y
    slope /= n * sum_x2 - sum_x**2
    alpha = -slope
    
    return alpha



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

def fit_alpha_linear_subsampling(series, frac=0.7, n_subsets=300, tail_frac_range=None, plot=True, return_scale=False):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    If return_scale is True, also returns the scale of the distribution.
    'tail_frac_range' defines what uniform range to draw from as a guess for where the tail starts
    in terms of the fraction of data used (from largest to smallest).
    '''
    
    # Set up lists
    _results = []
    
    # When no tail_frac_range is given a simple heuristic is used:
    if tail_frac_range is None:
        tail_frac_midle = get_tail_frac_guess(series)
        tail_frac_range = (tail_frac_midle/1.5, tail_frac_midle*1.5)
    
    # Subsample and fit
    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
        
        # Randomly choose a tail start from a uniform random distribution
        tail_frac = np.random.uniform(*tail_frac_range)
        
        _results.append(subsample.abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_scale=True))
        
    # Assemble into DataFrame
    results = pd.DataFrame(_results, columns=['Tail Exponent', 'Scale'])
    
    # Plot
    if plot:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        results['Tail Exponent'].hist(bins=10);
        plt.xlabel('Tail Exponent alpha');
        plt.title('Fit to log-log tail with random subsamples ({})'.format(series.name));
        plt.vlines(x=results['Tail Exponent'].mean(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', label='Mean ({:.2f})'.format(results['Tail Exponent'].mean()));
        plt.vlines(x=results['Tail Exponent'].median(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', linestyle='--', label='Median ({:.2f})'.format(results['Tail Exponent'].median()));
        plt.legend();
        
    # Construct result
    return results if return_scale else results.loc[:, ['Tail Exponent']]



#def fit_alpha_linear_subsampling(series, frac=0.7, n_subsets=300, tail_frac=0.1, plot=True, return_loc=False):
#    '''
#    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
#    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
#    If return_loc is True, also returns where the tail of the distribution is assumed to start.
#    'tail_frac' defines where the tail starts in terms of the fraction of data used (from largest to smallest).
#    '''
#    
#    # Set up lists
#    _results_both  = []
#    _results_left  = []
#    _results_right = []
#    
#    # Subsample and fit
#    for subsample in [series.sample(frac=frac) for i in range(n_subsets)]:
#        
#        _results_both.append(subsample.abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))
#        _results_left.append(subsample.where(subsample  < 0).abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))
#        _results_right.append(subsample.where(subsample >= 0).abs().agg(fit_alpha_linear, tail_frac=tail_frac, plot=False, return_loc=True))      
#        
#    # Assemble into DataFrame
#    alphas = pd.DataFrame.from_records(np.hstack([_results_both, _results_left, _results_right]), columns=pd.MultiIndex.from_product([['Both', 'Left', 'Right'], ['Tail Exponent', 'Location']]))    
#        
#    # Plot
#    if plot:
#        
#        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#        
#        fig.suptitle('Tail exponents for {} with random subsamples'.format(series.name))
#        
#        for idx, name in enumerate(['Both', 'Left', 'Right']):
#            
#            sns.histplot(data=alphas[(name, 'Tail Exponent')], color=['C7', 'C3', 'C0'][idx], stat='probability', bins=10, ax=ax[idx]);
#            ax[idx].set_title('Median = {:.1f} | Mean = {:.1f} ({})'.format(alphas[(name, 'Tail Exponent')].median(), alphas[(name, 'Tail Exponent')].mean(), ['both', 'left', 'right'][idx]));
#            ax[idx].set_xlabel('Tail exponent ({})'.format(['both', 'left', 'right'][idx]));
#            
#        plt.show();
#        
#        # Also plot locations if return_loc
#        if return_loc:
#        
#            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#        
#            fig.suptitle('Locations for {} with random subsamples'.format(series.name))
#            
#            for idx, name in enumerate(['Both', 'Left', 'Right']):
#                
#                sns.histplot(data=alphas[(name, 'Location')], color=['C7', 'C3', 'C0'][idx], stat='probability', bins=10, ax=ax[idx]);
#                ax[idx].set_title('Median = {:.1f} | Mean = {:.1f} ({})'.format(alphas[(name, 'Location')].median(), alphas[(name, 'Location')].mean(), ['both', 'left', 'right'][idx]));
#                ax[idx].set_xlabel('Location ({})'.format(['both', 'left', 'right'][idx]));
#                
#            plt.show();
#        
#    # Construct result
#    result = alphas if return_loc else alphas.loc[:, (slice(None), 'Tail Exponent')]
#    
#    return result



def max_likelihood_pareto_alpha(series, loc=0, min_samples=5, plot=False):
    '''
    Estimates the maximum likelihood for the tail exponent of a Pareto distribution.
    A location 'loc' can be specified which shifts the entire distribution.
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
            lambda x: 1 / (np.log(x - loc).mean() - np.log(x.min() - loc))
        )

        estimation.reset_index(drop=True).plot(grid=True, linestyle='--');
        plt.xlabel('Number of samples (descending order)');
        plt.ylabel('alpha');
        plt.title('Maximum Likelihood Estimation of Tail Exponent');
        plt.hlines(y=estimation.iloc[-1], xmin=0, xmax=len(cleaned_series));
        plt.legend(['Estimator', 'Result ({:.2f})'.format(estimation.iloc[-1])]);

        return estimation.iloc[-1]

    else:

        return 1 / (np.log(cleaned_series - loc).mean() - np.log(cleaned_series.min() - loc))


    
def max_likelihood_alpha(series, loc=0, min_samples=5, plot=False, tail_frac=None):
    '''
    Estimates the maximum likelihood for the tail exponent.
    A location 'loc' can be specified which shifts the entire distribution.
    If 'plot' visualizes the convergence of the estimator with increasing number of samples.
    'min_samples' determines the number of samples required. 
    'tail_frac' gives the fraction of data used (from largest to smallest) that is assumed to constitute the power law tail.
    '''
    
    # Drop NAs and take absolute values
    cleaned_series = abs(series.dropna())
    
    # When no tail_frac is given a simple heuristic is used:
    if tail_frac is None:
        tail_frac = get_tail_frac_guess(series)
    
    # Make sure we have enough data (relative to min_samples)
    assert len(cleaned_series) >= np.ceil(min_samples/tail_frac), 'Too few samples (n = {}). Get more data or decrease \'min_samples\'.'.format(len(cleaned_series))   
    
    # Take the largest 'tail_frac' samples (assumed to constitute the tail) and estimate alpha from maximum likelihood
    result = max_likelihood_pareto_alpha(
        series=cleaned_series.sort_values(ascending=False).iloc[:int(len(cleaned_series)*tail_frac)],
        loc=loc,
        min_samples=min_samples,
        plot=plot
    )
    
    return result



def get_tail_frac_guess(series):
    '''
    Uses a simple heuristic to estimate where the tail starts in terms of the fraction of samples in the tail.
    '''
    
    # Drop NAs
    cleaned_series = series.dropna()
    
    tail_frac = np.sqrt(1000/len(cleaned_series))*0.03
    
    return tail_frac
    


def max_likelihood_alpha_subsampling(series, loc=0, frac=0.7, n_subsets=300, tail_frac_range=None, plot=True):
    '''
    Estimates the tail parameter via maximum likelihood for alpha assuming a power law tail.
    A location 'loc' can be specified which shifts the entire distribution.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    'tail_frac_range' defines what uniform range to draw from as a guess for where the tail starts
    in terms of the fraction of data used (from largest to smallest).
    '''
    
    # Set up lists
    _results = []
    
    # When no tail_frac_range is given a simple heuristic is used:
    if tail_frac_range is None:
        tail_frac_midle = get_tail_frac_guess(series)
        tail_frac_range = (tail_frac_midle/1.5, tail_frac_midle*1.5)
    
    # Subsample and fit
    for subsample in [series.dropna().sample(frac=frac) for i in range(n_subsets)]:
        
        # Randomly choose a tail start from a uniform random distribution
        tail_frac = np.random.uniform(*tail_frac_range)
        
        _results.append(subsample.agg(max_likelihood_alpha, loc=loc, tail_frac=tail_frac, min_samples=2, plot=False))
        
    # Assemble into DataFrame
    results = pd.Series(_results)
        
    # Plot
    if plot:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        sns.histplot(x=results, stat='probability', binwidth=0.25, ax=ax);
        plt.xlabel('Tail Exponent alpha');
        plt.title('Maximum likelihood with random subsamples ({})'.format(series.name));
        plt.vlines(x=results.mean(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', label='Mean ({:.2f})'.format(results.mean()));
        plt.vlines(x=results.median(), ymin=0, ymax=plt.gca().get_ylim()[1], color='red', linestyle='--', label='Median ({:.2f})'.format(results.median()));
        plt.legend();
    
    return results



def hill_estimator(series, loc=0, plot=False, tail_frac=None):
    '''
    Estimates the tail exponent via the Hill Estimator.
    A location 'loc' can be specified which shifts the entire distribution.
    If 'plot' visualizes the convergence of the estimator with increasing number of samples.
    'tail_frac' gives the fraction of data used (from largest to smallest) that is assumed to constitute the power law tail.
    '''

    # Drop NAs and take absolute values
    cleaned_series = abs(series.dropna())

    # When no tail_frac is given a simple heuristic is used:
    if tail_frac is None:
        tail_frac = get_tail_frac_guess(series)

    # Make sure we have enough data (relative to min_samples)
    assert len(cleaned_series) >= 2, 'Too few samples (n = {}). Need at least 2 samples.'.format(len(cleaned_series))

    # Reorder data
    cleaned_series = cleaned_series.sort_values(ascending=False).reset_index(drop=True)

    # Hill estimator
    H_estimator = np.log(cleaned_series - loc).expanding(min_periods=2).mean() - np.log(cleaned_series - loc)

    # Get alpha from Hill estimator
    alpha_series = (1/H_estimator).dropna()
    alpha = alpha_series.iloc[int(len(alpha_series)*tail_frac) - 1]

    if plot:

        alpha_series.iloc[:int(len(alpha_series)*tail_frac) - 1].plot(grid=True, linestyle='--');
        plt.xlabel('Number of samples (descending order)');
        plt.ylabel('alpha');
        plt.title('Hill Estimator for Tail Exponent');
        plt.hlines(y=alpha, xmin=0, xmax=len(alpha_series.iloc[:int(len(alpha_series)*tail_frac) - 1]));
        plt.legend(['Estimator', 'Result ({:.2f})'.format(alpha)]);

    return alpha
