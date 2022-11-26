# A collection of helper functions returning parts of a distribution.
import pandas as pd



def get_left_tail(series):
    '''
    Returns the left side of a distribution
    '''
    
    return series.where(series  < 0).abs()



def get_right_tail(series):
    '''
    Returns the right side of a distribution
    '''
    
    return series.where(series >= 0).abs()



def get_both_tails(series):
    '''
    Returns abs values of both sides of a distribution
    '''
    
    return series.abs()



def shoulders_and_tails_t(df, loc, scale):
    '''
    Returns the location of the "shoulders" and the "tails" of a Student t distribution...
    ...with 'df' degrees of freedom, location 'loc', and 'scale'.
    '''
    
    shoulder = np.sqrt((5*df - np.sqrt((df+1) * (17*df+1)) + 1) / (df - 1)) * scale / np.sqrt(2)
    tail     = np.sqrt((5*df + np.sqrt((df+1) * (17*df+1)) + 1) / (df - 1)) * scale / np.sqrt(2)
    
    return -tail + loc, -shoulder + loc, shoulder + loc, tail + loc



def shoulders_and_tails_norm(mu, sigma):
    '''
    Returns the location of the "shoulders" and the "tails" of a normal distribution...
    ...with mean 'mu' and standard deviation 'sigma'.
    '''
    
    shoulder = np.sqrt(1/2 * (5 - np.sqrt(17))) * sigma
    tail     = np.sqrt(1/2 * (5 + np.sqrt(17))) * sigma
    
    return -tail + mu, -shoulder + mu, shoulder + mu, tail + mu
