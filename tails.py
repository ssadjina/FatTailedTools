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