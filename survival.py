# Methods to calculate survival functions
import pandas as pd
import numpy as np



def get_survival_function(series, drop_duplicates=True):
    '''
    Calculates a one-sided (abs) survival function for a Pandas Series.
    Returns a Pandas DataFrame with the columns "Values", X, and "P", P(x >= X), keeping the index (NAs dropped).
    'drop_duplicates' determines whether duplicates in the survival function are dropped or not.
    '''
    
    cleaned_series = abs(series.dropna())
    
    # Get values (from smallest to largest)
    survival = pd.DataFrame(np.linspace(0, cleaned_series.max(), len(cleaned_series)), index=cleaned_series.index, columns=['Values'])
    
    # Calculate probability for survival, that is, how many samples are above a certain value
    survival['P'] = survival['Values'].map(lambda x: (cleaned_series > x).mean())
    
    # Drop duplicates
    if drop_duplicates:
        survival = survival.drop_duplicates(subset='P', keep='last')
    
    return survival
