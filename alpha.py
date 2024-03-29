# A collection of various tools to help estimate and analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings



from FatTailedTools import survival
from FatTailedTools import plotting
from FatTailedTools import returns
from FatTailedTools import tails



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
    survival_tail = np.log10(survival_func.loc[survival_func['Values'] >= tail_start])
    
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



def fast_linear_fit(x, y):
    '''
    Optimized for speed to be used with, for example, subsampling and Monte Carlo simulations.
    Can be used to estimate the tail parameter by fitting a linear function to the log-log survival function.
    In that case, alpha = -slope
    '''

    # Calculate slope
    n = len(x)
    xy = x * y
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = xy.sum()
    sum_x2 = (x**2).sum()
    slope = n * sum_xy - sum_x * sum_y
    slope /= n * sum_x2 - sum_x**2

    # Calculate intercept
    intercept = sum_y - slope * sum_x
    intercept /= n

    return slope, intercept



def fit_alpha_linear_fast(series, threshold=0):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log survival function.
    Optimized for speed to be used with, for example, subsampling and Monte Carlo simulations.
    :param threshold: Gives the threshold for which the data is fitted.
    '''
    
    # Get survival function
    S = survival.get_survival_function(series)
    x = np.log10(S['Values'])
    y = np.log10(S['P'])

    # Select only the samples of the survival function which are >= threshold for the fit
    x_fit = x[S['Values'] >= threshold]
    y_fit = y[S['Values'] >= threshold]

    # Perform fast linear fit
    slope, intercept = fast_linear_fit(x_fit, y_fit)

    # Get alpha
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



def fit_alpha_and_scale_linear_subsampling(
        data,
        period_days,
        tail                 = 'both',
        transformation_func  = returns.get_log_returns,
        plot                 = True,
        frac                 = 0.9,
        min_samples          = 9,
        n_subsamples         = 300,
        n_fits_per_subsample = 30,
        n_debug_plots        = np.nan
):
    '''
    Estimates the tail exponent and the scale of a Student's t distribution for the left or right tail of the
    log returns of a time series.
    :param data:          Time series data passed as pandas.Series with a DateTimeIndex.
    :param period_days:   The time period used to generate the log returns in days passed as integer.
    :param tail:          Determines the side of the distribution ('left', 'right' or 'both' for absolute values).
    :param transformation_func: A function determining how to transform the time series. Uses returns.get_log_returns by
    default to calculate logarithmic returns. Use, e.g. returns.get_max_drawdowns to sue maximum drawdowns instead.
    :param plot:          Whether to plot the multivariate distribution over scales and tail coefficients.
    :param frac:          The fraction of all data to use for bootstrapping.
    :param min_samples:   Smallest number of samples required for the linear fit.
    :param n_subsamples:  Number of times parameters are estimated using bootstrap.
    :param n_fits_per_subsample:  Number of thresholds to check to determine optimal threshold (and help estimate parameters).
    :param n_debug_plots: Produces a number of 'n_debug_plots' plots to help check for errors in the fitting routine.
    :return:              The estimated tail exponent ('estimated_tail_exponent'), the estimated scale ('estimated_scale') of a Student's t
    distribution with the estimated tail exponent, a pandas DataFrame holding all results ('df_results'), and a dictionary ('dists')
    of the form {parameter: (dist, params)} that holds two fitted univariate lognormal distributions ('dist') and their parameters
    ('params') for the scale ('Scale') and for the tail coefficient ('Tail Coefficient'), respectively.
    '''

    # Sanity checks
    assert isinstance(data, pd.Series), '\'data\' needs to be a Pandas Series with DateTimeIndex.'
    assert isinstance(data.index, pd.DatetimeIndex), '\'data\' needs to have a DateTimeIndex.'
    assert isinstance(period_days, int)
    assert period_days >= 1, '\'period_days\' needs to be larger than 0.'
    assert min_samples >= 3, '\'min_samples\' need to be at least 3.'
    min_samples_total = int(period_days * 1/frac * 2 * 2 * (min_samples + 1))
    assert len(data.dropna()) >= min_samples_total, 'Not enough samples in \'data\'. {} required, {} found.'.format(min_samples_total, len(data.dropna()))
    assert tail in ['left', 'right', 'both'], '\'tail\' must be either \'left\', \'right\', or \'both\'.'

    # Set up
    results = []

    # Get function to use to get left or right tail, or both (absolute)
    if tail == 'both':
        get_tail = tails.get_both_tails
    elif tail == 'left':
        get_tail = tails.get_left_tail
    else:
        get_tail = tails.get_right_tail

    # Subsample to address various types of uncertainty when estimating the tail exponent:
    #    1. Use different time shifts/origins (if period > 1) to adress the uncertainty wrt. choosing the "right" shift/origin.
    #    2. Use bootstrapping to address the fact that we always have incomplete data.
    #    3. Repeat the linear fit to the log-log survival function with different thresholds for where the tail starts.
    for i in range(n_subsamples):

        # --------------------------------------------------------------------------------------------
        # Handle uncertainty wrt. the "correct" origin / time shift.

        # Randomly select a time shift/origin
        time_shift = np.random.choice(range(period_days))

        # Apply the transformation over 'period' and using a shift 'time_shift'.
        # By default, this will be log returns.
        series     = transformation_func(data, periods='{}d'.format(period_days), offset=time_shift).dropna()

        # --------------------------------------------------------------------------------------------
        # Use bootstrapping to include the uncertainty wrt. to the data.
        subsample = series.sample(frac=frac, replace=True)

        # --------------------------------------------------------------------------------------------
        # Calculate the survival function

        # Get the desired tail of the distribution
        subsample = get_tail(subsample).dropna()

        # Get survival function
        S = survival.get_survival_function(subsample)
        abs_series = S['Values']
        if len(abs_series) < 3 + min_samples:
            warnings.warn('Not enough data to select optimal threshold. Continuing. Get more data, or try decreasing \'min_samples\', or increasing \'frac\'.')
            continue
        x = np.log10(abs_series)
        y = np.log10(S['P'])

        # --------------------------------------------------------------------------------------------
        # Handle uncertainty wrt. the "correct" threshold, that is, where the tail starts

        # Choose highest threshold such that at least 'min_samples' are left beyond (for the linear fit)
        threshold_max  = abs_series.iloc[-min_samples]
        assert threshold_max > 0

        # Choose lowest threshold as median of all samples below
        threshold_min  = abs_series.iloc[:-min_samples].median()

        # Get all samples that lie in between:
        thresholds     = abs_series.loc[abs_series.between(threshold_min, threshold_max, inclusive='both')]

        # If there are more than 'n_fits_per_subsample', use a linear spacing instead
        if len(thresholds) > n_fits_per_subsample:
            thresholds = np.linspace(threshold_min, threshold_max, n_fits_per_subsample)
        assert len(thresholds) >= 3

        # Set up lists to store results of the linear fits
        fitted_tail_exponents = []
        fitted_scales         = []
        fitted_MSE            = []

        # Iterate through the treshold range
        for threshold in thresholds:

            # Select only the samples of the survival function which are >= threshold for the fit
            x_fit = x[abs_series >= threshold]
            y_fit = y[abs_series >= threshold]

            # Fit linear regression model using OLS
            slope, intercept = fast_linear_fit(x_fit, y_fit)

            # Get tail exponent
            tail_exponent = -slope

            # Get scale
            scale = 10**(- intercept / slope)

            # Get mean squared error (MSE)
            y_pred = intercept + x_fit * slope
            MSE    = np.mean((y_pred - y_fit)**2)

            # Store results
            fitted_tail_exponents.append(tail_exponent)
            fitted_scales.append(scale)
            fitted_MSE.append(MSE)

        # Prepare array of MSE results and get lowest achieved MSE for any of the fits
        fitted_MSE              = np.array(fitted_MSE)
        assert len(fitted_MSE) >= 1
        MSE_min                 = fitted_MSE.min()
        assert MSE_min          > 0

        # Determine the best tail exponent and threshold
        # This is done via exponential weighting of the results wrt. the MSE
        weights                 = np.exp(-fitted_MSE / MSE_min)
        best_fit_tail_exponent  = np.average(fitted_tail_exponents, weights=weights)
        best_fit_scale          = np.average(fitted_scales,         weights=weights)
        best_fit_threshold      = np.average(thresholds,            weights=weights)

        # --------------------------------------------------------------------------------------------
        # Fit Student's t distribution using the estimated tail exponent and location = 0

        # To be able to do this, the data is mirrored around 0
        t_series = pd.concat([-abs_series, abs_series])

        # Fit and extract scale
        try:
            t_params = t.fit(t_series, f0=best_fit_tail_exponent, floc=0)
        except FitError:
            # If fitting routine fails, return np.nan as scale
            print('WARNING: Fit of Student\'s t to data failed for alpha={:.2f} and threshold={:.4f}.'.format(
                best_fit_tail_exponent,
                best_fit_threshold
            ))
            t_params = [np.nan] * 3
        t_scale  = t_params[-1]

        # --------------------------------------------------------------------------------------------
        # Collect results
        results.append([
            best_fit_tail_exponent,
            best_fit_scale,
            t_scale,
            MSE_min,
            best_fit_threshold,
            time_shift,
        ])

        # --------------------------------------------------------------------------------------------
        # Create a debug plot (with a maximum number of 'n_debug_plots' plots)
        if np.mod(i, n_subsamples//n_debug_plots) == 0:

            # Set up subplots
            fig, ax = plt.subplot_mosaic(
                [
                    [2, 1],
                    [0, 1]
                ],
                figsize=(8, 3), gridspec_kw={'width_ratios': [1, 2]}, constrained_layout=True
            );

            # Plot Survival Function
            x_plot_range = [threshold_min/10, 10**np.max(x)*2]
            y_plot_range = [10**np.min(y)/2, 1]
            ax[1].scatter(10**x, 10**y, s=1);
            ax[1].set_xscale('log');
            ax[1].set_yscale('log');
            ax[1].set_title('Survival Function');
            ax[1].set_xlabel('Threshold');
            ax[1].axvspan(best_fit_threshold, x_plot_range[1], *y_plot_range, alpha=0.1, color='r');

            # Plot Student's t fit
            x_plot = np.geomspace(*x_plot_range, 500);
            p = 1 - np.sign(x_plot) * t.cdf(x_plot, *t_params) - np.sign(-x_plot) * t.cdf(-x_plot, *t_params);
            ax[1].plot(x_plot, p, c='C3');

            # Plot linear fit using best threshold
            slope, intercept = fast_linear_fit(
                x[abs_series >= best_fit_threshold],
                y[abs_series >= best_fit_threshold]
            )
            ax[1].plot(x_plot, 10**(intercept + np.log10(x_plot) * slope), c='C3', linestyle=':');
            ax[1].set_xlim(x_plot_range);
            ax[1].set_ylim(y_plot_range);
            ax[1].legend(['Data', 'Tail', 'Fit (t)', 'Fit (linear)']);

            # Plot weighting
            ax[0].fill_between(thresholds, weights, 0, alpha=0.6);
            ax[0].set_ylabel('Weighting');
            ax[0].set_xlabel('Threshold');

            # Plot tail exponent
            ax[2].fill_between(x=thresholds, y1=fitted_tail_exponents, y2=best_fit_tail_exponent, alpha=0.6);
            ax[2].set_ylabel('alpha');
            ax[2].hlines(best_fit_tail_exponent, *ax[2].get_xlim(), color='C3');

            # Set axes
            for axis in [0, 1, 2]:
                ax[axis].grid(which='both');
                ax[axis].vlines(best_fit_threshold, *ax[axis].get_ylim(), color='C3', linestyle='--');

            plt.show();

    # --------------------------------------------------------------------------------------------
    # Assemble results

    # Construct DataFrame to return
    df_results = pd.DataFrame(results, columns=['Tail Exponent', 'Scale (Pareto)', 'Scale', 'MSE', 'Threshold', 'Time Shift'])

    # Add tail coefficient (inverse alpha)
    df_results['Tail Coefficient'] = 1 / df_results['Tail Exponent']

    # --------------------------------------------------------------------------------------------
    # If 'plot', plot the results for the tail coefficient and the scale
    if plot:

        # Plot histogram and fits and return lognormal fits to tail coefficient and scale
        dists = plot_alpha_and_scale_fit_hist(df_results);

    else:

        # If not given, fit results to
        # a. two univariate log normal distributions
        # b. a multivariate log normal distribution
        dists = {}

        # Fit result to multivariate lognormal distribution
        multivariate_dist, lognorm_params_x, lognorm_params_y = fit_multivariate_lognormal(
            df_results['Scale'],
            df_results['Tail Coefficient']
        )
        dists.update({('Scale', 'Tail Coefficient'): multivariate_dist})

        # Fit univariate distributions
        dists.update({'Scale':            (lognorm(*lognorm_params_x), lognorm_params_x)})
        dists.update({'Tail Coefficient': (lognorm(*lognorm_params_y), lognorm_params_y)})

    # --------------------------------------------------------------------------------------------
    # Calculate the final best estimate for the tail exponent and the scale.
    # We assume that everything (tail exponent, tail coefficient, scale) is log normal distributed.
    # Note that the log normal approximates the normal for small standard deviations.

    # Get mean for tail coefficient from the multivariate lognormal
    estimated_tail_coefficient = transform_normal_to_lognormal(dists[('Scale', 'Tail Coefficient')].mean[1], *dists['Tail Coefficient'][1])

    # The estimated tail exponent is the inverse of the estimated tail coefficient
    estimated_tail_exponent    = 1. / estimated_tail_coefficient

    # Get mean for scale from the multivariate lognormal
    estimated_scale            = transform_normal_to_lognormal(dists[('Scale', 'Tail Coefficient')].mean[0], *dists['Scale'][1])

    return estimated_tail_exponent, estimated_scale, df_results, dists



from scipy.stats import lognorm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def plot_alpha_and_scale_fit_hist(
        df,
        x     = 'Scale',
        y     = 'Tail Coefficient',
        dists = None,
        **kwargs
):
    '''
    Convenience function to plot a 2D histogram of tail coefficient and scale after a fit with fit_alpha_and_scale_linear_subsampling.
    :param df: Pandas DataFrame that holds the results from the fitting procedure fit_alpha_and_scale_linear_subsampling().
    :param x: Label of column in df to be used for the x values. Default is the Student's t scale.
    :param y: Label of column in df to be used for the y values. Default is the tail coefficient (inverse alpha).
    :param dists: A dict holding the univariate distributions for x and y and their parameters. Default is None in which case x and y are fitted to univariate lognormal distributions each.
    :param kwargs: Arguments that can be passed to seaborn.JointGrid().
    :return: A dict holding the univariate distributions for x and y and their parameters, as well as a multivariate distribution. If none are given via the 'dists' argument, x and y are fitted accordingly.
    '''

    # Initialize the figure
    g = sns.JointGrid(x=df[x], y=df[y], marginal_ticks=False, **kwargs)

    # Create an inset legend for the histogram colorbar
    cax = g.figure.add_axes([.75, .60, .02, .2])

    # Add the joint and marginal histogram plots
    g.plot_joint(
        sns.histplot, bins='fd', stat='density', cbar=True, cbar_ax=cax
    )
    g.plot_marginals(sns.histplot, bins='fd', stat='density', element='step');

    # If not given, fit results to
    # a. two univariate log normal distributions
    # b. a multivariate log normal distribution
    if dists is None:

        dists = {}

        # Fit result to multivariate lognormal distribution
        multivariate_dist, lognorm_params_x, lognorm_params_y = fit_multivariate_lognormal(df[x], df[y])
        dists.update({(x, y): multivariate_dist})

        # Fit univariate distributions
        dists.update({x: (lognorm(*lognorm_params_x), lognorm_params_x)})
        dists.update({y: (lognorm(*lognorm_params_y), lognorm_params_y)})

    # Add plots of the univariate distributions to the marginals
    x_plot = np.linspace(df[x].min(), df[x].max(), 100)
    y_plot = np.linspace(df[y].min(), df[y].max(), 100)
    _ = g.ax_marg_x.plot(x_plot, dists[x][0].pdf(x_plot), c='C3');
    _ = g.ax_marg_y.plot(dists[y][0].pdf(y_plot), y_plot, c='C3');

    # Add contour plot of the multivariate distribution to the
    # Set up the meshgrid
    X, Y = np.meshgrid(x_plot, y_plot)

    # Calculate the PDF at each point on the grid
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = transform_lognormal_to_normal(X, *dists[x][1])
    pos[:, :, 1] = transform_lognormal_to_normal(Y, *dists[y][1])
    Z = dists[(x, y)].pdf(pos)

    # Plot the contours
    g.ax_joint.contour(X, Y, Z, cmap=LinearSegmentedColormap.from_list(
        'cmap_transparent_to_C3', [(*to_rgb('C3'), 0), (*to_rgb('C3'), 1)]))

    # Add best guesses (= means of the multivariate lognormal)
    x_guess = transform_normal_to_lognormal(dists[(x, y)].mean[0], *dists[x][1])
    y_guess = transform_normal_to_lognormal(dists[(x, y)].mean[1], *dists[y][1])
    g.refline(x=x_guess, y=y_guess, c='C3');
    print(x_guess, y_guess)

    plt.show();

    return dists



from scipy.stats import multivariate_normal

def transform_lognormal_to_normal(x, s, loc, scale):
    '''
    Transforms lognormally distributed data to normally distributed data using the parameters obtained from a scipy.stats.lognormal fit
    :param x: Array holding the lognormal distributed data.
    :param s: Shape parameter of the lognormal.
    :param loc: Location parameter of the lognormal.
    :param scale: Scale parameter of the lognormal.
    :return: The transformed array
    '''

    return np.log(s * (x - loc)) - np.log(scale)



def transform_normal_to_lognormal(log_x, s, loc, scale):
    '''
    Transforms normally distributed data to lognormally distributed data using the parameters obtained from a scipy.stats.lognormal fit
    :param x: Array holding the normal distributed data.
    :param s: Shape parameter of the lognormal.
    :param loc: Location parameter of the lognormal.
    :param scale: Scale parameter of the lognormal.
    :return: The transformed array.
    '''

    return np.exp(log_x + np.log(scale)) / s + loc



def fit_multivariate_lognormal(x, y):
    '''
    Fits a multivariate normal distribution to the log transform of the two vectors x and y.
    :param x: Array holding the x values.
    :param y: Array holding the y values
    :return: (dist, lognorm_params_x, lognorm_params_y); where 'dist' is a scipy.stats.multivariate_normal object fitted to the (log of the) data, and 'lognorm_params_{x/y}' are the parameters obtained from fitting the data to scipy.stats.lognormal.
    '''

    # Get lognormal fit parameters to be able to scale the data properly
    lognorm_params_x = lognorm.fit(x, loc=np.mean(x), scale=np.exp(np.log(x).mean()))
    lognorm_params_y = lognorm.fit(y, loc=np.mean(y), scale=np.exp(np.log(y).mean()))

    # Get log transforms with the correct scaling
    log_x = transform_lognormal_to_normal(x, *lognorm_params_x)
    log_y = transform_lognormal_to_normal(y, *lognorm_params_y)

    # Extract the mean and standard deviation of the log transformed data
    mean1 =  log_x.mean()
    std1  =  log_x.std()
    mean2 =  log_y.mean()
    std2  =  log_y.std()

    # Define the covariance matrix
    rho = np.corrcoef(log_x, log_y)[0, 1]
    cov = [[std1**2, rho*std1*std2], [rho*std1*std2, std2**2]]

    # Create a multivariate_normal object with the means and covariance of the log transformed data
    dist = multivariate_normal(mean=[mean1, mean2], cov=cov)

    return dist, lognorm_params_x, lognorm_params_y



def sample_multivariate_lognormal(dist, lognorm_params_x, lognorm_params_y, size):
    '''
    Fits a multivariate normal distribution to the log transform of the two vectors x and y.
    Returns 'size' number of samples drawn from the fitted distribution.
    '''

    # Generate samples from the fitted distribution
    log_data = dist.rvs(size=size)

    # Transform back
    data = np.vstack([
        transform_normal_to_lognormal(log_data[:, 0], *lognorm_params_x),
        transform_normal_to_lognormal(log_data[:, 1], *lognorm_params_y)
    ]).T

    return data



def fit_and_sample_multivariate_lognormal(x, y, size):

    # Fit data
    dist, lognorm_params_x, lognorm_params_y = fit_multivariate_lognormal(x, y)

    # Sample
    data = sample_multivariate_lognormal(dist, lognorm_params_x, lognorm_params_y, size)

    return data



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



def share_of_total(percentage, alpha, return_value_nan=np.nan):
    '''
    Calculates the share of the total that a certain top percentage of a population holds collectively.
    :param percentage: The percentage of the largest individuals in a population to consider.
    :param alpha: The tail exponent of the Pareto distribution.
    :param return_value_nan: The value to return when alpha is not > 1.
    :return: The share of the total. If alpha is not > 1, 'return_value_nan' is returned.
    '''

    result = np.exp((1 - 1 / alpha) * np.log(percentage))

    return np.where(alpha > 1, result, return_value_nan)
