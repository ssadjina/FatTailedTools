# A collection of helper functions returning parts of a distribution.
import pandas as pd
import numpy as np



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



import distributions

def get_shoulders_and_tails_from_JointDistribution(joint_dist):
    '''
    Returns the location of the "shoulders" and the "tails" of the two Student t distributions in a 'JointDistribution'.
    '''
    
    params_left, params_right = distributions.get_params_from_JointDistribution(joint_dist)
    
    tail_start_left, shoulder_point_left, _, _   = shoulders_and_tails_t(*params_left )
    _, _, shoulder_point_right, tail_start_right = shoulders_and_tails_t(*params_right)
    
    return tail_start_left, shoulder_point_left, shoulder_point_right, tail_start_right
