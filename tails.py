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



def rebalance_tails(series, method='oversampling'):
    '''
    Rebalances the tails to make sure the left and the right tail have equal amounts of (notnull) samples.
    Uses random over- or undersampling with replacement to match minority and majority sides.
    :param series: Pandas Series that holds the samples (both tails).
    :param method: Either 'oversampling' (draw samples from minority side to match majority side in size) or
    'undersampling' (draw samples from majority side to match minority side in size), both with replacement.
    :return: Returns the rebalances series as a Pandas Series object.
    '''

    # Sanity check
    assert method in ['oversampling', 'undersampling']

    # Split samples into left and right tails
    series_left  = -get_left_tail(series).dropna()
    series_right =  get_right_tail(series).dropna()

    # Sanity checks
    assert len(series_left)  > 0
    assert len(series_right) > 0

    # Identify minority and majority side
    if len(series_left) < len(series_right):
        series_minority, series_majority = series_left, series_right
    else:
        series_minority, series_majority = series_right, series_left

    # Resample
    if method == 'oversampling':
        # Use random oversampling with replacement to match the sample size of the minority side with the majority side
        series_minority = series_minority.sample(n=len(series_majority), replace=True)
    elif method == 'undersampling':
        # Use random undersampling with replacement to match the sample size of the majority side with the minority side
        series_majority = series_majority.sample(n=len(series_minority), replace=True)
    else:
        assert False

    # Assemble new rebalanced series
    series_rebalanced = pd.concat([series_minority, series_majority], axis=0)

    # Sanity check
    assert (series_rebalanced < 0).dropna().sum() == (series_rebalanced > 0).dropna().sum()

    return series_rebalanced



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



from FatTailedTools import distributions

def get_shoulders_and_tails_from_JointDistribution(joint_dist):
    '''
    Returns the location of the "shoulders" and the "tails" of the two Student t distributions in a 'JointDistribution'.
    '''
    
    params_left, params_right = distributions.get_params_from_JointDistribution(joint_dist)
    
    tail_start_left, shoulder_point_left, _, _   = shoulders_and_tails_t(*params_left )
    _, _, shoulder_point_right, tail_start_right = shoulders_and_tails_t(*params_right)
    
    return tail_start_left, shoulder_point_left, shoulder_point_right, tail_start_right
