from scipy.stats import rv_continuous
import numpy as np
import matplotlib.pyplot as plt



# Define new class that allows us to assemble a hybrid PDF with different left and right sides
class JointDistribution(rv_continuous):
    def __init__(self, dist_left, dist_right):
        self.dist_left  = dist_left
        self.dist_right = dist_right
        super().__init__()
        
    # PDF
    def _pdf(self, x):
        return np.where(x < 0, self.dist_left.pdf(x), self.dist_right.pdf(x))
        
    # CDF
    def _cdf(self, x):
        return np.where(x < 0, self.dist_left.cdf(x), self.dist_right.cdf(x))
        
    # Survival Function
    def _sf(self, x):
        return np.where(x < 0, self.dist_left.sf(x), self.dist_right.sf(x))
    
    # Inverse CDF
    def _ppf(self, p):
        # If the argument is < 50%, we use the left tail, otherwise the right
        return np.where(p < 0.5, self.dist_left.ppf(p), self.dist_right.ppf(p))
    
    # Inverse Survival Function
    def _isf(self, p):
        # If the argument is < 50%, we use the left tail, otherwise the right
        return np.where(p < 0.5, self.dist_left.isf(p), self.dist_right.isf(p))
    
    
    
def get_params_from_JointDistribution(joint_dist):
    '''
    Retrieves the parameters for the left and the right distributions, respectively, in 'JointDistribution'.
    '''
    
    # Collect distributions
    dist_left  = joint_dist.dist_left
    dist_right = joint_dist.dist_right
    
    # Get parameters
    params_left  = list(dist_left.dist._parse_args( *dist_left.args,  **dist_left.kwds ))
    params_right = list(dist_right.dist._parse_args(*dist_right.args, **dist_right.kwds))
    
    # Flatten arguments where necessary
    if len(params_left[ 0]) > 0:
        params_left[ 0] = params_left[ 0][0]
    if len(params_right[0]) > 0:
        params_right[0] = params_right[0][0]
    
    return params_left, params_right



def get_alphas_from_JointDistribution(joint_dist):
    '''
    Retrieves the tail exponents alpha for the left and the right distributions, respectively, in 'JointDistribution'.
    '''
    
    params_left, params_right = get_params_from_JointDistribution(joint_dist)
    
    return params_left[0], params_right[0]



def fit_dist_parameter_over_time(series, guess_exponent=0.5, guess_log_range=0.35):
    '''
    Estimates the relationship between distribution parameters (scale or tail exponent) and time as given by 
    the Pandas Series 'series'.
    Returns a factor and an exponent, such that the estimated parameter = factor * days_to_expiration ** exponent,
    as well as the MSE error of the fit.
    'guess_exponent' is the initial guess for the exponent.
    'guess_log_range' can be adjusted to widen or narrow the (log) range over which the guesses are varied for optimization.
    '''
    
    min_error = np.inf
    
    for exponent in np.linspace(np.exp(-guess_log_range), np.exp(guess_log_range), 300) * guess_exponent:
        
        # Get fit error per element in the series
        diff  = np.log(series) - (np.log((series/np.power(series.index, exponent)).mean()) + exponent * np.log(series.index))
        
        # Get MSE
        error = (diff**2).mean()
        
        if error < min_error:
            min_error     = error
            best_exponent = exponent
    
    # Get best factor based on best found exponent
    best_factor = (series/np.power(series.index, best_exponent)).mean()
            
    print('Best fit with exponent of {:.3f} and factor of {:.5f} (MSE = {:3f})'.format(best_exponent, best_factor, min_error))
    
    series.plot(label='Estimates', linestyle='', color='C0', marker='.', markersize=4);
    plt.xlabel('Days to expiration');
    plt.ylabel('Parameter');
    plt.grid(which='both');
    plt.plot(series.index, best_factor*np.power(series.index, best_exponent), color='C7', linestyle='--', label='{:.5f}*t**{:.3f}'.format(best_factor, best_exponent));
    plt.legend();
    
    return best_exponent, best_factor, min_error
    