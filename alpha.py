# A collection of various tools to help analyze the tail exponent.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from FatTailedTools.plotting import plot_survival_function

def fit_alpha_linear(series, tail_start_sigma=2):
    '''
    Estimates the tail parameter by fitting a linear function to the log-log tail of the survival function.
    'tail_start_sigma' defines where the tail starts in terms of standard deviations.
    '''
    
    # Get survival function values
    survival, ax = plot_survival_function(series, tail_zoom=True)
    
    # Estimate tail start (= everything beyond 'tail_start_sigma' sigmas)
    tail_start = tail_start_sigma*series.std()
    
    # Get tail
    survival_tail = np.log10(survival.loc[survival['Values'] >= tail_start].iloc[:-1])
    
    # Fit the tail
    tail_fit = np.polyfit(survival_tail['Values'], survival_tail['P'], 1)

    # Plot the fit
    ax.plot(10**survival_tail['Values'], 10**(survival_tail['Values']*tail_fit[0] + tail_fit[1]), 'r');
    ax.legend(['Fit', 'Data']);
    plt.title('Tail exponent fitted to tail (alpha = {:.2f})'.format(-tail_fit[0]));
    
    return -tail_fit[0]
