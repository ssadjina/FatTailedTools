# A collection of various visual tools to help analyze and study fat-tailed data.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



from FatTailedTools import survival
from FatTailedTools import kappa_metric



def plot_histgrams(series, distribution=None):
    '''
    Plots two histograms of a Pandas Series, one on linear axis and one on logarithmic axis.
    A scipy.stats distribution object can be passed via "distribution" to also plot the PDF.
    '''
    fig, ax = plt.subplots(2, 1, figsize=(15, 10));
    
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
    
    N1t_gains = 1 + (cleaned_series.iloc[1:] > cleaned_series.iloc[1:].shift(1).cummax()).cumsum()
    N1t_losses = 1 + (cleaned_series.iloc[1:] < cleaned_series.iloc[1:].shift(1).cummin()).cumsum()
    
    N1t_gains.plot(color='C0', figsize=(15, 8));
    N1t_losses.plot(color='C3');
    plt.title('Gumbel record test');
    
    x_ticks = list(range(len(cleaned_series)))
    sns.lineplot(x=x_ticks, y=(1 / (np.array(x_ticks) + 1)).cumsum(), color='k', linestyle='--');
    
    plt.legend(['Maxima', 'Minima', 'Expected']);
    plt.xlabel('Samples');
    

    
def max_sum_plot(series):
    '''
    Plots maximum-to-sum plots for all four moments of a Pandas Series.
    '''
    
    cleaned_series = abs(series.dropna())
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10));
    
    for idx, axis in enumerate(ax.reshape(-1)): 
        
        sns.lineplot(
            data=((cleaned_series**(idx + 1)).cummax())/((cleaned_series**(idx + 1)).cumsum()),
            ax=axis
        ).set_title('{}. Moment'.format(idx + 1));
        axis.set_xlabel('Samples');
    
    
    fig.suptitle('Maximum-to-Sum Plot');
    
    

def plot_survival_function(series, tail_zoom=False, distribution=None, figsize=(10, 5), point_size=5, title_annotation=None):
    '''
    Plots a one-sided (abs) survival function for a Pandas Series, and returns the survival data itself ("survival") and the figure object.
    If "tail_zoom", the tail part of the distribution is visualized.
    A scipy.stats distribution object can be passed via "distribution" to also plot the PDF.
    "title_annotation" allows to add text in parenthesis to the title of the plot.
    '''
    
    # Get survival function
    survival = get_survival_function(series)
    
    cleaned_series = abs(series.dropna())
    
    # Plot
    plot_title = 'Survival Function'
    if title_annotation is not None:
        plot_title += ' ({})'.format(title_annotation)
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=figsize);
        sns.scatterplot(data=survival, x='Values', y='P', s=point_size, ax=ax).set_title(plot_title);
    
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
        x = np.logspace(-4, cleaned_series.max(), 1000)
        p = 1 - np.sign(x) * distribution[0].cdf(x, *distribution[1]) - np.sign(-x) * distribution[0].cdf(-x, *distribution[1])
        
        # Plot
        with sns.axes_style('whitegrid'):
            sns.lineplot(x=x, y=p,  color='C3', ax=ax);
            ax.legend(['Fitted distribution ({})'.format(distribution[0].name), 'Samples']);
        
    # Set axes
    ax.set_xscale('log');
    ax.set_yscale('log');
    ax.set_xlim([int(tail_zoom) * cleaned_series.std(), None] if tail_zoom is not False else [cleaned_series.loc[cleaned_series > 0].min()/2, None]);
    ax.set_ylim([1/len(cleaned_series)/2, 1/3] if tail_zoom else [1/len(cleaned_series)/2, 1]);
    ax.set_xlabel('X');
    ax.set_ylabel('P(|x| > X)');
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor')
        
    return survival, ax



def plot_twosided_survival_function(series, tail_zoom=False, distribution_right=None, distribution_left=None, figsize=(10, 5), point_size=5, title_annotation=None):
    '''
    Plots a two-sided (left and right tail) survival function for a Pandas Series, and returns the survival data itself ("survival") and the figure object.
    If "tail_zoom", the tail part of the distribution is visualized.
    A scipy.stats distribution objects can be passed via "distribution_right" and "distribution_left" to also plot the PDFs of the tails.
    "title_annotation" allows to add text in parenthesis to the title of the plot.
    '''
    
    cleaned_series = series.dropna()
    
    series_left  = cleaned_series.loc[cleaned_series <  0]
    series_right = cleaned_series.loc[cleaned_series >= 0]
    
    # Get values (from smallest to largest)
    survival = pd.DataFrame(np.linspace(0, abs(cleaned_series).max(), len(cleaned_series)), index=cleaned_series.index, columns=['Values'])
    
    # Calculate probability for survival, that is, how many samples are above a certain value
    survival['P_left']  = survival['Values'].map(lambda x: (-series_left > x).mean())
    survival['P_right'] = survival['Values'].map(lambda x: (series_right > x).mean())
    
    # Drop duplicates
    survival = survival.drop_duplicates(subset=['P_left', 'P_right'], keep='last')
    
    # Plot
    plot_title = 'Survival Function'
    if title_annotation is not None:
        plot_title += ' ({})'.format(title_annotation)
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=figsize);
        sns.scatterplot(data=survival, x='Values', y='P_right', color='C0', s=point_size, ax=ax).set_title(plot_title);
        sns.scatterplot(data=survival, x='Values', y='P_left',  color='C3', s=point_size, ax=ax).set_title(plot_title);
        ax.legend(['Right', 'Left']);
    
    if (distribution_right is not None) | (distribution_left is not None):
        
        # Get analytical distributions
        x = np.logspace(-4, cleaned_series.max(), 1000)
        p_right = 1 - np.sign(x) * distribution_right[0].cdf(x, *distribution_right[1]) - np.sign(-x) * distribution_right[0].cdf(-x, *distribution_right[1])
        p_left  = 1 - np.sign(x) * distribution_left[0].cdf(x, *distribution_left[1]) - np.sign(-x) * distribution_left[0].cdf(-x, *distribution_left[1])
        
        # Plot
        with sns.axes_style('whitegrid'):
            sns.lineplot(x=x, y=p_right,  color='C0', ax=ax);
            sns.lineplot(x=x, y=p_left,  color='C3', ax=ax);
            
            ax.legend(['Samples (right)', 'Samples (left)', 'Fitted distribution (right) ({})'.format(distribution_right[0].name), 'Fitted distribution (left) ({})'.format(distribution_left[0].name)]);
        
    # Set axes
    ax.set_xscale('log');
    ax.set_yscale('log');
    ax.set_xlim([int(tail_zoom) * cleaned_series.std(), None] if tail_zoom is not False else [cleaned_series.loc[cleaned_series > 0].min()/2, None]);
    ax.set_ylim([1/len(cleaned_series)/2, 1/3] if tail_zoom else [1/len(cleaned_series)/2, 1]);
    ax.set_xlabel('X');
    ax.set_ylabel('P(|x| > X)');
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor')
        
    return survival, ax



def plot_empirical_kappa_n(series, n_bootstrapping=10000, n_values=list(range(20, 601, 20))):
    '''
    Plots the empirical kappa_n metric using "n_bootstrapping" bootstrapping samples n in "n_values".
    See Nassim Taleb's "Statistical Consequences of Fat Tails".
    '''
    
    cleaned_series = series.dropna()
    
    assert len(series.dropna()) >= 600, 'Need at least 600 samples'
    
    series_right = cleaned_series.loc[cleaned_series >= 0]
    series_left  = cleaned_series.loc[cleaned_series <  0]

    y_right = [kappa_n(series_right, n, n_bootstrapping=n_bootstrapping) for n in n_values]
    y_left  = [kappa_n(series_left , n, n_bootstrapping=n_bootstrapping) for n in n_values]
    
    
    with sns.axes_style('whitegrid'):
        fig, ax = plt.subplots(1, figsize=(10, 5));
        
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
    
    cleaned_series = series.dropna()
    
    lindy = pd.DataFrame(np.linspace(0.01, max(abs(cleaned_series)), len(cleaned_series)), index=cleaned_series.index, columns=['k'])
    
    # Calculate the expectation value for -X conditional on -X < k, divided by k.
    lindy['Lindy'] = lindy['k'].map(lambda k: ((-cleaned_series > k) * (-cleaned_series)).mean() / ((-cleaned_series > k).mean()) / k)
    
    # Plot
    fig, ax = plt.subplots(1, figsize=(10, 5));
    sns.lineplot(data=lindy, x='k', y='Lindy', color='C3', ax=ax).set_title('\"Lindy Measure\"');
    
    return lindy
