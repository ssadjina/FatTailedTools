# A collection of various visual tools to help analyze and study fat-tailed data.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

FIG_SIZE=(10, 5)



from FatTailedTools import survival
from FatTailedTools import kappa_metric
from FatTailedTools import alpha
from FatTailedTools import tails



def plot_histograms(series, distribution=None):
    '''
    Plots two histograms of a Pandas Series, one on linear axis and one on logarithmic axis.
    A scipy.stats distribution object can be passed via "distribution" to also plot the PDF.
    '''
    fig, ax = plt.subplots(2, 1, figsize=(FIG_SIZE[0], FIG_SIZE[0]));

    # Plot linear
    sns.histplot(data=series, stat='probability' if distribution is None else 'density', kde=True if distribution is None else False, ax=ax[0]);

    # Plot log transformed
    sns.histplot(data=series, stat='probability' if distribution is None else 'density', ax=ax[1], log_scale=(False, True if distribution is None else False));

    # Draw distribution PDF if given
    if distribution is not None:
        x_range = np.linspace(series.min(), series.max(), 1000)
        sns.lineplot(x=x_range, y=distribution[0].pdf(x_range, *distribution[1]),  color='C3', ax=ax[0]);
        sns.lineplot(x=x_range, y=distribution[0].pdf(x_range, *distribution[1]),  color='C3', ax=ax[1]);
        ax[1].set_yscale('log');



def plot_gumbel_test(series):
    '''
    Plots the cummulative count for how many times a sample in a Pandas Series constituted a new minima or maxima (Gumbel record test).
    Regardless of the underlying distribution, the probability that a sample is larger than any of the N samples seen before is 1/N.
    '''

    cleaned_series = series.dropna()

    # Check if left/right tails are given
    bool_left_tail  = (cleaned_series < 0).any()
    bool_right_tail = (cleaned_series > 0).any()

    # Right tail (maxima)
    if bool_right_tail:
        N1t_gains = 1 + (cleaned_series.iloc[1:] > cleaned_series.iloc[1:].shift(1).cummax()).cumsum()
        N1t_gains.plot(color='C0', figsize=FIG_SIZE, label='Maxima');

    # Left tail (minima)
    if bool_left_tail:
        N1t_losses = 1 + (cleaned_series.iloc[1:] < cleaned_series.iloc[1:].shift(1).cummin()).cumsum()
        N1t_losses.plot(color='C3', label='Minima');

    # Plot the theoretically expected (which is independent of the distribution)
    x_ticks = list(range(len(cleaned_series)))
    H_t = (1 / (np.array(x_ticks) + 1)).cumsum()
    sns.lineplot(x=cleaned_series.index, y=H_t, color='k', linestyle='--', label='Expected');

    plt.title('Gumbel record test');
    plt.legend();
    plt.xlabel('Samples');



def max_sum_plot(series):
    '''
    Plots maximum-to-sum plots for all four moments of a Pandas Series.
    '''

    # Prepare data and take absolute values
    cleaned_series = abs(series.dropna())

    # Calculate max-to-sum for the first four moments
    plots = []

    for moment in range(1, 5):

        plots.append(
            ((cleaned_series**moment).cummax())/((cleaned_series**moment).cumsum())
        )

    # Assemble plots
    plots = pd.concat(plots, axis=1)

    # Rename columns
    plots.columns = ['{}. Moment'.format(moment) for moment in range(1, 5)]

    # Plot everything in a 2x2 subplot grid
    plots.plot(subplots=True, layout=(2, 2), figsize=(FIG_SIZE[0], FIG_SIZE[0]), grid=True, color='C0', title='Maximum-to-Sum Plot');



def plot_survival_function(series, tail_zoom=False, distribution=None, figsize=FIG_SIZE, point_size=5, title_annotation=None):
    '''
    Plots a one-sided (abs) survival function for a Pandas Series, and returns the survival data itself ("survival_func") and the axis object.
    If "tail_zoom", the tail part of the distribution is visualized.
    A scipy.stats distribution object can be passed via "distribution" to also plot the PDF.
    "title_annotation" allows to add text in parenthesis to the title of the plot.
    '''

    # Get survival function
    survival_func = survival.get_survival_function(series)

    cleaned_series = abs(series.dropna())

    # Plot
    plot_title = 'Survival Function'
    if title_annotation is not None:
        plot_title += ' ({})'.format(title_annotation)
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=figsize);
        sns.scatterplot(data=survival_func, x='Values', y='P', s=point_size, ax=ax).set_title(plot_title);

    # Also plot distribution, if provided
    # ===================================
    # Cummulative Distribution Function (CDF) F. Gives P that x <= X, for x, X >= 0
    # F_X(x) = P(x <= X)

    # Survival Function S. Gives P that x > X, for x, X >= 0
    # S_X(x) = 1 - F_X(x) = P(x > X)

    # Now, we have two-sided PDFs (with negative CDF values), so we need to figure out P for |x| > X, for X >= 0
    # P(|x| > X) = P(x > X)               + P(-x > X)
    # P(|x| > X) = Sign(x) * [1 - F_X(x)] + Sign(-x) * [1 - F_X(-x)]
    # P(|x| > X) = 1 - Sign(x) * F_X(x) - Sign(-x) * F_X(-x)

    if distribution is not None:

        # Get analytical distribution
        x = np.logspace(-4, np.log10(2*cleaned_series.max()), 1000)
        p = 1 - np.sign(x) * distribution[0].cdf(x, *distribution[1]) - np.sign(-x) * distribution[0].cdf(-x, *distribution[1])

        # Plot
        with sns.axes_style('whitegrid'):
            sns.lineplot(x=x, y=p,  color='C3', ax=ax);
            ax.legend(['Samples', 'Fitted distribution ({})'.format(distribution[0].name)]);

    # Set axes
    ax.set_xscale('log');
    ax.set_yscale('log');
    ax.set_xlim([int(tail_zoom) * cleaned_series.std(), None] if tail_zoom is not False else [cleaned_series.loc[cleaned_series > 0].min()/2, None]);
    ax.set_ylim([1/len(cleaned_series)/2, 1/3] if tail_zoom else [1/len(cleaned_series)/2, 1]);
    ax.set_xlabel('X');
    ax.set_ylabel('P(|x| > X)');
    ax.grid(visible=True, which='both')

    return survival_func, ax



def plot_twosided_survival_function(series, distribution=None, figsize=FIG_SIZE, point_size=5, title_annotation=None, log_threshold=None, n_subsets=1, subset_frac=0.7):
    '''
    Plots a two-sided survival function for a Pandas Series ("series") using a linear-log axis for the values,
    and a logit axis for the probabilities to show the entire distribution in one plot.
    Returns the survival data itself ("survival_func") and the axis object.
    A scipy.stats distribution object can be passed via "distribution" to also plot the complementary CDF (survival function).
    "title_annotation" allows to add text in parenthesis to the title of the plot.
    "log_threshold" determines the transition from linear to log on the x axis and can be adjusted for better viewing.
    "n_subsets" control the number of randomly drawn subsets using a fraction "subset_frac" (with replacement) of "series".
    '''

    # Clean and prepare data
    cleaned_series = series.dropna()

    # Prepare figure and axes
    plot_title = 'Survival Function'
    if title_annotation is not None:
        plot_title += ' ({})'.format(title_annotation)
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=figsize);

    # Get series maximum
    series_max = cleaned_series.abs().max()

    # Threshold where the linear scale goes over to logarithmic
    if log_threshold is None:
        log_threshold = cleaned_series.abs().quantile(0.5)

    for idx in range(n_subsets):

        subset = cleaned_series if n_subsets == 1 else cleaned_series.sample(frac=subset_frac, replace=True)

        # Get survival function
        survival_func = survival.get_twosided_survival_function(subset)

        # Plot
        sns.scatterplot(data=survival_func, x='Values', y='P', s=point_size, color='C0', alpha=1./n_subsets, ax=ax, label='Samples' if n_subsets==1 else '_nolegend_');

    ax.grid(visible=True, which='both');
    ax.set_xscale('symlog', linthresh=log_threshold, linscale=0.5, subs=range(2,10));
    ax.set_yscale('logit');
    ax.set_ylim([1./len(cleaned_series)/2., 1.-1./len(cleaned_series)/2.]);
    ax.set_xlim([-2*series_max, 2*series_max]);
    ax.set_xlabel('X');
    ax.set_ylabel('P(x > X)');
    ax.set_title(plot_title);

    # Also fit and plot if 'dist' is given
    if distribution is not None:
        #dist_params = distribution.fit(cleaned_series)

        # Get analytical distribution
        x = np.hstack([np.linspace(0, log_threshold, 500), np.logspace(np.log10(log_threshold), np.log10(2*series_max), 500)]);
        x= sorted(np.hstack([-x, x]));
        p = 1.-distribution[0].cdf(x, *distribution[1])

        # Plot
        with sns.axes_style('whitegrid'):
            sns.lineplot(x=x, y=p,  color='C3', ax=ax, label='Fitted distribution ({})'.format(distribution[0].name));
            ax.legend();

    return survival_func, ax



from scipy.stats import norm

def graphical_alpha_estimation(series, loc=0, frac=0.7, n_subsets=30, plot=True, max_samples_per_subset=300, sd=2):
    '''
    Graphical estimation of the tail exponent.
    Uses Hill Estimator and linear fit to the log-log survival plot.
    A location 'loc' can be specified which shifts the entire distribution.
    If 'plot' visualizes the convergence of the estimator with increasing number of samples.
    Uses 'n_subsets' subsamples to average results over subsets with a fraction 'frac' of samples kept.
    'max_samples_per_subset' allows to restrict the number of samples per subset.
    'sd' gives the standard deviations used in the plots (sd = 2 corresponds to 95% CI).
    '''

    # Minimum number of samples required
    MIN_N_SAMPLES = 3

    # Drop NAs and take absolute values
    cleaned_series = (series.dropna() - loc).abs().reset_index(drop=True)
    len_cleaned_series = len(cleaned_series)

    # Make sure we have enough data (relative to min_samples)
    assert len_cleaned_series >= MIN_N_SAMPLES, 'Too few samples (n = {}). Need at least {} samples.'.format(len(cleaned_series), MIN_N_SAMPLES)

    # Set up arrays to save results
    #results_hill = []
    results_zipf = []
    warnings.warn('Hill estimator temporarily deactivated due to a potential bug in the calculations.')

    # Choose a "safe" fraction of samples to draw for each subset that accounts for an upper bound 'max_samples_per_subset'
    safe_frac = min(max_samples_per_subset / len_cleaned_series, frac)

    # Subsample and calculate estimators
    for idx, subsample in enumerate([cleaned_series.sample(frac=safe_frac, replace=True) for i in range(n_subsets)]):

        # Reorder data
        ordered_series = subsample.sort_values(ascending=False).reset_index(drop=True)

        # Linear fit on Zipf plot
        zipf_estimator = ordered_series.expanding(min_periods=MIN_N_SAMPLES).apply(alpha.fit_alpha_linear_fast)

        # Hill estimator
        #hill_estimator = 1. / ((np.log(ordered_series).expanding(min_periods=MIN_N_SAMPLES).mean() - np.log(ordered_series)).dropna())

        # Collect results
        #results_hill.append(hill_estimator)
        results_zipf.append(zipf_estimator)

    # Order results to get back to original ordered samples
    #results_hill = pd.concat(results_hill, axis=1)
    results_zipf = pd.concat(results_zipf, axis=1)
    #results_hill.columns = range(n_subsets)
    results_zipf.columns = range(n_subsets)

    # Assemble DataFrame for Zipf estimator
    df_longform_zipf = pd.DataFrame(results_zipf.stack()).reset_index()
    df_longform_zipf.columns = ['Order Statistics', 'Experiment', 'Estimator']
    df_longform_zipf['Order Statistics'] = (df_longform_zipf['Order Statistics'] / safe_frac).astype('int')
    df_longform_zipf['Threshold'] = cleaned_series.sort_values(ascending=False).reset_index(drop=True).loc[df_longform_zipf['Order Statistics']].values

    ## Assemble DataFrame for Hill estimator
    #df_longform_hill = pd.DataFrame(results_hill.stack()).reset_index()
    #df_longform_hill.columns = ['Order Statistics', 'Experiment', 'Estimator']
    #df_longform_hill['Order Statistics'] = (df_longform_hill['Order Statistics'] / safe_frac).astype('int')
    #df_longform_hill['Threshold'] = cleaned_series.sort_values(ascending=False).reset_index(drop=True).loc[df_longform_hill['Order Statistics']].values

    if plot:

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=FIG_SIZE, sharex=False, sharey=False)

        MIN_THRESHOLD = cleaned_series.quantile(0.25)

        # Plot estimators
        sns.lineplot(data=df_longform_zipf.loc[df_longform_zipf['Threshold'] >= MIN_THRESHOLD], x='Threshold', y='Estimator', errorbar=('sd', sd), ax=ax[0], label='Zipf plot fit');
        #sns.lineplot(data=df_longform_hill.loc[df_longform_hill['Threshold'] >= MIN_THRESHOLD], x='Threshold', y='Estimator', errorbar=('sd', sd), ax=ax[0], label='Hill estimator');
        ax[0].set_xscale('log');
        ax[0].set_ylabel('alpha (CI = {:.0%})'.format(1.-2.*(1.-norm.cdf(sd))));
        ax[0].set_xlim([MIN_THRESHOLD, df_longform_zipf['Threshold'].max()]);
        ax[0].set_ylim([0, 6]);
        ax[0].grid(visible=True, which='both');
        ax[0].legend();

        # Plot survival function log-log plot
        survival_func = survival.get_survival_function(cleaned_series.dropna().abs())
        ax[1].plot(survival_func['Values'], survival_func['P'], linestyle='', color='C0', marker='.', markersize=4)
        ax[1].set_xscale('log');
        ax[1].set_yscale('log');
        ax[1].set_xlim([MIN_THRESHOLD, cleaned_series.max()*1.1]);
        ax[1].grid(visible=True, which='both');
        ax[1].set_xlabel('Threshold');
        ax[1].set_ylabel('Survival probability');

    else:
        return None, df_longform_zipf  # df_longform_hill, df_longform_zipf



def plot_empirical_kappa_n(series, n_bootstrapping=10000, n_values=list(range(20, 601, 20))):
    '''
    Plots the empirical kappa_n metric using "n_bootstrapping" bootstrapping samples n in "n_values".
    See Nassim Taleb's "Statistical Consequences of Fat Tails".
    '''

    cleaned_series = series.dropna()

    assert len(series.dropna()) >= 600, 'Need at least 600 samples'

    series_right = cleaned_series.loc[cleaned_series >= 0]
    series_left  = cleaned_series.loc[cleaned_series <  0]

    y_right = [kappa_metric.kappa_n(series_right, n, n_bootstrapping=n_bootstrapping) for n in n_values]
    y_left  = [kappa_metric.kappa_n(series_left , n, n_bootstrapping=n_bootstrapping) for n in n_values]


    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=FIG_SIZE);

        sns.scatterplot(x=n_values, y=y_right, color='C0', s=9, ax=ax);
        sns.scatterplot(x=n_values, y=y_left , color='C3', s=9, ax=ax);

    ax.set_title('Empirical Kappa');
    ax.set_xlabel('n');
    ax.set_ylabel('k_n');

    ax.legend([
        'Right',
        'Left'
    ]);



def plot_lindy_test(series):
    '''
    Plots the Lindy test from a Pandas Series.
    See Nassim Taleb's "Statistical Consequences of Fat Tails".
    '''

    # Prepare data
    cleaned_series_abs = series.abs().dropna()
    cleaned_series     = -cleaned_series_abs

    # Get guess for where tail starts
    threshold_min = cleaned_series_abs.sort_values(ascending=False).iloc[int(alpha.get_tail_frac_guess(cleaned_series_abs)*len(cleaned_series))]

    # Get guess for largest threshold to use
    threshold_max = max(cleaned_series_abs)/2

    # Set up thresholds
    lindy = pd.DataFrame(np.linspace(threshold_min, threshold_max, len(cleaned_series)), columns=['k'])

    # Calculate the expectation value for -X conditional on -X < k, divided by k.
    lindy['E(-X | -X > k) / k'] = lindy['k'].map(lambda k: ((cleaned_series_abs > k) * (cleaned_series_abs)).mean() / ((cleaned_series_abs > k).mean()) / k)

    # Plot
    fig, ax = plt.subplots(1, figsize=FIG_SIZE);
    sns.lineplot(data=lindy, x='k', y='E(-X | -X > k) / k', color='C3', ax=ax).set_title('\"Lindy Test\"');
    ax.grid(visible=True, which='both');



def mean_excess_plot(series):
    '''
    Plots mean excess of 'series' with varying thresholds.
    '''

    # Clean and take absolute values
    cleaned_series = series.dropna().abs()

    # Set up list
    _x = cleaned_series.sort_values(ascending=True).values
    _y = []

    for u in _x:
        # To get the conditional expectation for X - u given that X > u, we need to:
        # 1. Integrate P(X)*(X - u) from u to inf
        # 2. Normalize by P(X > u) = integral of P(X) from u to inf; This is the survival function S(u)

        _part1 = ((cleaned_series > u) * (cleaned_series - u)).mean()
        _part2 = ((cleaned_series > u)).mean()

        me = _part1 / _part2

        _y.append(me)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE);
    plt.plot(_x, _y, linestyle='', marker='.');
    plt.title('Mean Excess Plot');
    plt.xlabel('Threshold');
    plt.ylabel('Mean Excess');
    plt.show()



def plot_lorenz_curve(series, ascending=True, tail_exponent=None):
    '''
    Plots the Lorenz curve of the absolute values of samples.
    A Lorenz curve shows the cumulative fraction of the total quantity against the cumulative fraction
    of the ordered population.
    :param series: A Pandas Series object containing the data. Note that the absolute values will be used.
    :param ascending: If true, samples are ordered from smallest to largest,
    corresponding to the traditional Lorenz curve.
    If false, from largest to smallest.
    :param tail_exponent: The tail exponent that the series is assumed to follow. If not None, plot the theoretically
    expected Lorenz curve.
    :return: The data used for the plot as a Pandas Series.
    '''

    # Prepare data
    abs_series_sorted = series.dropna().abs().sort_values(ascending=ascending)

    # Replace index with the required percentage share
    series_new_index = abs_series_sorted.reset_index(drop=True)
    series_new_index.index += 1                            # +1 so we start counting at 1 (not zero) samples
    series_new_index.index /= len(series_new_index) / 100  # Get percentage share

    # Plot data
    (series_new_index.cumsum()/series_new_index.sum() * 100).plot(color='C0', figsize=FIG_SIZE, label='Observed');
    plt.xlim([series_new_index.index.min(), series_new_index.index.max()]);
    plt.ylim([0, 100]);
    plt.ylabel('Cumulative fraction of total [%]');
    plt.xlabel('Cumulative fraction of population [%]');
    plt.title('Lorenz curve');

    # If a tail exponent 'alpha' is given, also plot the theoretical curve
    if tail_exponent is not None:
        x_plot = np.linspace(series_new_index.index.min(), 100, 1000)
        plt.plot(x_plot, (ascending + (1 - 2 * ascending) * alpha.share_of_total(
            percentage = ascending + (1 - 2 * ascending) * x_plot / 100,
            alpha=tail_exponent
        )) * 100, color='C3', label='Expected (Tail exponent = {:.2f})'.format(tail_exponent));

        plt.legend();

    return series_new_index
