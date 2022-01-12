# Methods to calculate survival functions
import pandas as pd
import numpy as np



def get_survival_function(series, inclusive=True):
    '''
    Calculates a (one-sided) survival function from (the absolute values of) a Pandas Series 'series'.
    Returns a Pandas DataFrame with the columns "Values", X, and "P", P(x >= X), keeping the index (and NAs dropped).
    If 'inclusive', P(x >= X) is used and the largest data point is plotted. Else, P(x > X) is used and the largest data point is not plotted. The latter is consistent with, e.g., seaborn.ecdfplot(complementary=True).
    '''
    
    # Take absolute values and drop NAs
    abs_series = series.dropna().abs()
    
    # Set up DataFrame and sort values from largest to smallest
    survival = pd.DataFrame(abs_series, index=abs_series.index, columns=['Values']).sort_values(by='Values', ascending=True)
    
    # Determine whether we compare with '>=' or with '>'
    if inclusive:
        func = lambda x: (survival['Values'] >= x).mean()
    else:
        func = lambda x: (survival['Values'] > x).mean()
    
    # Get survival probabilities
    survival['P'] = survival['Values'].apply(func)
    
    return survival
