"""
Metrics comparing predicted and recorded firing rates

All metrics in this module are computed separately for each cell, but averaged
across the sample dimension (axis=0).
"""
from functools import wraps

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

__all__ = ['cc', 'rmse', 'fev', 'CC', 'RMSE', 'FEV', 'np_wrap',
           'root_mean_squared_error', 'correlation_coefficient',
           'fraction_of_explained_variance','correlation_coefficient_distribution','fraction_of_explainable_variance_explained','fraction_of_explainable_variance_explained_K']


def correlation_coefficient(obs_rate, est_rate):    # (y_true, y_pred)
    """Pearson correlation coefficient"""
    if obs_rate.shape[-1]>1:
        x_mu = obs_rate - tf.experimental.numpy.nanmean(obs_rate, axis=-1, keepdims=True)
        x_std = K.std(obs_rate, axis=-1, keepdims=True)
        y_mu = est_rate - tf.experimental.numpy.nanmean(est_rate, axis=-1, keepdims=True)
        y_std = K.std(est_rate, axis=-1, keepdims=True)
        cc = tf.experimental.numpy.nanmean(x_mu * y_mu, axis=-1, keepdims=True) / (x_std * y_std)
        
    else:
        x_mu = obs_rate - K.mean(obs_rate, axis=0, keepdims=True)
        x_std = K.std(obs_rate, axis=0, keepdims=True)
        y_mu = est_rate - K.mean(est_rate, axis=0, keepdims=True)
        y_std = K.std(est_rate, axis=0, keepdims=True)
        cc = K.mean(x_mu * y_mu, axis=0, keepdims=True) / (x_std * y_std)
        
    return cc



# def correlation_coefficient(obs_rate, est_rate):    # (y_true, y_pred)
#     """Pearson correlation coefficient"""
    
#     x_mu = obs_rate - tf.experimental.numpy.nanmean(obs_rate, axis=-1, keepdims=True)
#     x_std = K.std(obs_rate, axis=-1, keepdims=True)
#     y_mu = est_rate - tf.experimental.numpy.nanmean(est_rate, axis=-1, keepdims=True)
#     y_std = K.std(est_rate, axis=-1, keepdims=True)
#     return tf.experimental.numpy.nanmean(x_mu * y_mu, axis=-1, keepdims=True) / (x_std * y_std)



def mean_squared_error(obs_rate, est_rate):
    """Mean squared error across samples"""
    if obs_rate.shape[-1]>1:
        mse_val = tf.experimental.numpy.nanmean(K.square(est_rate - obs_rate), axis=-1, keepdims=True)
    else:
        mse_val = K.mean(K.square(est_rate - obs_rate), axis=0, keepdims=True)
    
    return mse_val

def root_mean_squared_error(obs_rate, est_rate):
    """Root mean squared error"""
    return K.sqrt(mean_squared_error(obs_rate, est_rate))


def fraction_of_explained_variance(obs_rate, est_rate):
    """Fraction of explained variance

    https://wikipedia.org/en/Fraction_of_variance_unexplained
    """
    
    if obs_rate.shape[-1]>1:
        fev_val = 1.0 - mean_squared_error(obs_rate, est_rate) / K.var(est_rate, axis=-1, keepdims=True)
    else:
        fev_val = 1.0 - mean_squared_error(obs_rate, est_rate) / K.var(est_rate, axis=0, keepdims=True)
        
    return fev_val

# def fraction_of_explained_variance(obs_rate, est_rate):
#     """Fraction of explained variance

#     https://wikipedia.org/en/Fraction_of_variance_unexplained
#     """
#     return 1.0 - mean_squared_error(obs_rate, est_rate) /np.nanvar(obs_rate, axis=0, keepdims=True)

def fraction_of_explainable_variance_explained(obs_rate, est_rate,obs_noise):
    resid = obs_rate - est_rate
    mse_resid = np.mean(resid**2,axis=0)
    # mse_resid = np.mean(resid**2,axis=-1)
    var_test = np.var(est_rate,axis=0)
    # var_test = np.var(est_rate,axis=-1)
    fev_allUnits = 1 - ((mse_resid - obs_noise)/(var_test-obs_noise))
    fev_median = np.median(fev_allUnits)
    fev_std = np.std(fev_allUnits)

    return fev_allUnits

def fraction_of_explainable_variance_explained_K(obs_rate, est_rate,obs_noise=0):
    resid = obs_rate - est_rate
    mse_resid = tf.experimental.numpy.nanmean(resid**2,axis=0)
    var_test = K.var(obs_rate,axis=0)
    fev_allUnits = 1 - ((mse_resid - obs_noise)/(var_test-obs_noise))
    # fev_median = K.median(fev_allUnits)
    # fev_std = np.std(fev_allUnits)

    return fev_allUnits



def correlation_coefficient_distribution(obs_rate,est_rate):
    x_mu = obs_rate - np.mean(obs_rate, axis=0)
    x_std = np.std(obs_rate, axis=0)
    y_mu = est_rate - np.mean(est_rate, axis=0)
    y_std = np.std(est_rate, axis=0)
    cc_allUnits = np.mean(x_mu * y_mu,axis=0) / (x_std * y_std)
    return cc_allUnits


def np_wrap(func):
    """Converts the given keras metric into one that accepts numpy arrays

    Usage
    -----
    corrcoef = np_wrap(cc)(y, yhat)     # y and yhat are numpy arrays
    """
    @wraps(func)
    def wrapper(obs_rate, est_rate):
        with tf.Session() as sess:
            # compute the metric
            yobs = tf.placeholder(tf.float64, obs_rate.shape)
            yest = tf.placeholder(tf.float64, est_rate.shape)
            metric = func(yobs, yest)

            # evaluate using the given values
            feed = {yobs: obs_rate, yest: est_rate}
            return sess.run(metric, feed_dict=feed)
    return wrapper


# aliases
cc = CC = correlation_coefficient
rmse = RMSE = root_mean_squared_error
# fev = FEV = fraction_of_explained_variance
fev = FEV = fraction_of_explained_variance

