#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:53:23 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import numpy as np
import os
import h5py
import math
from scipy.stats import zscore
# from model import utils_si
# from model.performance import model_evaluate
import re
import json
# from tqdm import tqdm
import gc
import random

from collections import namedtuple


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        # arr = arr[:,np.newaxis,:,:]
        return array
                 
def unroll_data(data,time_axis=0,rolled_axis=1):
    rgb = data[0]
    rgb = np.concatenate((rgb,data[1:,data.shape[1]-1,:]),axis=0)
    # rgb = np.concatenate((rgb,data[-1:,0,:]),axis=0)
    return rgb
        
def isintuple(x,name):
    t = type(x)
    attrs = getattr(t, '_fields', None)
    if name in attrs:
        return True
    else:
        return False

def h5_tostring(arr):
    new_arr = np.empty(arr.shape[0],dtype='object')
    for i in range(arr.shape[0]):
        rgb = arr[i].tostring().decode('ascii')
        rgb = rgb.replace("\x00","")
        new_arr[i] = rgb

    return (new_arr)


def prepare_data_cnn2d(data,filt_temporal_width,idx_unitsToTake,num_chunks=1,MAKE_LISTS=False):
    
    if data.X.ndim==5:       # if the data has multiple stims and trials
        X = data.X
        y = data.y


        X_rgb = X.reshape(X.shape[0],X.shape[1],X.shape[2],-1,order='A')
        y_rgb = y.reshape(y.shape[0],y.shape[1],-1,order='A')

        
        if filt_temporal_width==1: # the case when filt_temporal_width=0
            X_rgb = X_rgb[:,None,:,:,:]
        else:
            X_rgb = rolling_window(X_rgb,filt_temporal_width,time_axis=0)
            y_rgb = y_rgb[filt_temporal_width:]
        
        
        X_list = []
        y_list = []
        i=0
        for i in range(X_rgb.shape[-1]):
            rgb = list(X_rgb[:,:,:,:,i])
            X_list = X_list + rgb
            
            rgb = list(y_rgb[:,:,i])
            y_list = y_list + rgb
        
        X = X_list
        y = y_list

        del X_rgb, y_rgb
    
    else:
        
        if filt_temporal_width>0:
            X = rolling_window(data.X,filt_temporal_width,time_axis=0)   
            y = data.y[:,idx_unitsToTake]
            y = y[filt_temporal_width:]
            if isintuple(data,'spikes')==True:
                spikes = data.spikes[:,idx_unitsToTake]
                spikes = spikes[filt_temporal_width:]
                
            if isintuple(data,'y_trials')==True:
                y_trials = data.y_trials[:,idx_unitsToTake]
                y_trials = y_trials[filt_temporal_width:]

        else:
            X = np.expand_dims(data.X,axis=1)
            y = data.y[:,idx_unitsToTake]
            if isintuple(data,'y_trials')==True:
                y_trials = data.y_trials[:,idx_unitsToTake]

        if X.ndim==5:       # if the data has multiple stims
            X_rgb = np.moveaxis(X,0,-1)
            X_rgb =  X_rgb.reshape(X_rgb.shape[0],X_rgb.shape[1],X_rgb.shape[2],-1)
            X_rgb = np.moveaxis(X_rgb,-1,0)
            
            y_rgb = np.moveaxis(y,0,-1)
            y_rgb = y_rgb.reshape(y_rgb.shape[0],-1)
            y_rgb = np.moveaxis(y_rgb,-1,0)
            
            X = X_rgb
            y = y_rgb
            
            del X_rgb, y_rgb
            
        if MAKE_LISTS==True:    # if we want to arrange samples as lists rather than numpy arrays
            X = list(X)
            y = list(y)
            if isintuple(data,'y_trials')==True:
                y_trials = list(y_trials)

            if isintuple(data,'spikes')==True:
                spikes = list(spikes)

    
    data_vars = ['X','y','spikes','y_trials']
    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data = namedtuple('Exptdata',dataDict)
    data=data(**dataDict)
    
    del X, y
    return data

    
"""
This is a very messy function as I've tried to evolve it for different datasets. 
I never got to tidy it up. Once i tidy it up, its only a few lines. So maybe don't try to understand it yet.
"""
def load_h5Dataset(fname_data_train_val_test,LOAD_TR=True,LOAD_VAL=True,LOAD_ALL_TR=False,nsamps_val=-1,nsamps_train=-1,nsamps_test=-1,RETURN_VALINFO=False,
                   idx_train_start=0,VALFROMTRAIN=False,LOADFROMBOOL=False,dtype='float16'):     # LOAD_TR determines whether to load training data or not. In some cases only validation data is required
    FLAG_VALFROMTRAIN=False
    f = h5py.File(fname_data_train_val_test,'r')
    t_frame = np.array(f['parameters']['t_frame'])
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])
    f_keys = list(f.keys())

    if LOADFROMBOOL == True:
        if nsamps_train.dtype=='bool':
            idx = np.where(nsamps_train)
        else:
            idx = nsamps_train
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_train = Exptdata_spikes(X,y,spikes)
        return data_train

        
    
    # some loading parameters
    if nsamps_val==-1 or nsamps_val==0:
        idx_val_start = 0
        idx_val_end = -1
    else:
        nsamps_val = int((nsamps_val*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_val_start = 0
        idx_val_end = idx_val_start+nsamps_val
        
    if nsamps_test==-1 or nsamps_test==0:
        idx_test_start = 0
        idx_test_end = -1
    else:
        nsamps_test = int((nsamps_test*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_test_start = f['data_test']['X'].shape[0]-nsamps_test
        idx_test_end = idx_test_start+nsamps_test

        
    idx_train_start = int((idx_train_start*60*1000)/(t_frame))    # mins to frames
    if nsamps_train==-1 or nsamps_train==0 :
        # idx_train_start = 0
        idx_train_end = -1
        # idx_data = np.arange(idx_train_start,np.array(f['data_train']['y'].shape[0]))
    else:
        LOAD_ALL_TR = False
        if nsamps_train <1000:  # i.e. if this is in time, else it is in samples
            nsamps_train = int((nsamps_train*60*1000)/t_frame)
        # idx_train_start = 0
        idx_train_end = idx_train_start+nsamps_train
        # idx_data = np.arange(idx_train_start,idx_train_end)
    
    
    # Training data
    if LOAD_TR==True:   # only if it is requested to load the training data
        regex = re.compile(r'data_train_(\d+)')
        dsets = [i for i in f_keys if regex.search(i)]
        if len(dsets)>0:    # if the dataset is split into multiple datasets
            if LOAD_ALL_TR==True:   # concatenate all datasets into one
                X = np.array([]).reshape(0,f[dsets[0]]['X'].shape[1],f[dsets[0]]['X'].shape[2])
                y = np.array([]).reshape(0,f[dsets[0]]['y'].shape[1])
                spikes = np.array([]).reshape(0,f[dsets[0]]['spikes'].shape[1])
                for i in dsets:
                    rgb = np.array(f[i]['X'])
                    X = np.concatenate((X,rgb),axis=0)
                    
                    rgb = np.array(f[i]['y'])
                    y = np.concatenate((y,rgb),axis=0)
                    
                    rgb = np.array(f[i]['spikes'])
                    spikes = np.concatenate((spikes,rgb),axis=0)

                X = X.astype('float32')
                y = y.astype('float32')
                spikes = spikes.astype('float32')
                
                
                                
                data_train = Exptdata_spikes(X,y,spikes)
            else:            # just pick the first dataset
                X = np.array(f[dsets[0]]['X'][idx_train_start:idx_train_end],dtype='float32')
                y = np.array(f[dsets[0]]['y'][idx_train_start:idx_train_end],dtype='float32')
                spikes = np.array(f[dsets[0]]['spikes'][idx_train_start:idx_train_end],dtype='float32')
                data_train = Exptdata_spikes(X,y,spikes)
                # data_train = Exptdata(X,y)
            
        else:   # if there is only one dataset
            if idx_train_end!=-1:
                if nsamps_val==0:   # for backwards compat
                    nsamps_val = int((0.3*60*1000)/t_frame)
                
                # Take data offset by start time. Take validation and test data from center of training data
                bool_idx_train = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_val = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_test = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_val_test = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                
                nsamps_test = int(nsamps_val/4)
                nsamps_val_test = nsamps_val+nsamps_test
                nsamps_train_val_test = nsamps_train+nsamps_val+nsamps_test

                mid =  int(nsamps_train_val_test/2) + idx_train_start       # mid point of train_val_test
                bool_idx_val_test[mid-int(nsamps_val_test/2):mid] = True
                bool_idx_val_test[mid:mid+int(nsamps_val_test/2)] = True
                bool_idx_train[idx_train_start:idx_train_start+nsamps_train_val_test] = True
                bool_idx_train[bool_idx_val_test] = False
                if VALFROMTRAIN == True:
                    idx_val_start = np.where(bool_idx_val_test)[0][0]
                    bool_idx_val[idx_val_start:idx_val_start+nsamps_val] = True
                    bool_idx_test[idx_val_start+nsamps_val:idx_val_start+nsamps_val+nsamps_test]  = True
                    
                    assert(sum(bool_idx_train&bool_idx_val)<2)
                    assert(sum(bool_idx_train&bool_idx_test)<2)
                    assert(sum(bool_idx_val&bool_idx_test)<2)
                    
                    data_val_info = dict(nsamps_train=nsamps_train,nsamps_val=nsamps_val,nsamps_test=nsamps_test,
                                         bool_idx_train=bool_idx_train,bool_idx_val=bool_idx_val,bool_idx_test=bool_idx_test)
                
                    # plt.plot(bool_idx_train);plt.plot(bool_idx_val);plt.plot(bool_idx_test);plt.xlim([260000,270000]);plt.show();
                
            else:
                bool_idx_train = np.ones(f['data_train']['X'].shape[0],dtype='bool')
            
            idx = np.where(bool_idx_train)
            X = np.array(f['data_train']['X'][idx],dtype='float32')
            y = np.array(f['data_train']['y'][idx],dtype='float32')
            if 'spikes' in f['data_train']:
                spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
                data_train = Exptdata_spikes(X,y,spikes)
            else:
                data_train = Exptdata(X,y)
            

    else:
        data_train = None
        
    # Validation data
    if VALFROMTRAIN==False:
        if LOAD_VAL==True:
            regex = re.compile(r'data_val_(\d+)')
            dsets = [i for i in f_keys if regex.search(i)]
            if len(dsets)>0:
                d = 0
                X = np.array(f[dsets[d]]['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f[dsets[d]]['y'][idx_val_start:idx_val_end],dtype='float32')
                spikes = np.array(f[dsets[d]]['spikes'][idx_val_start:idx_val_end],dtype='float32')
                data_val = Exptdata_spikes(X,y,spikes)
    
                      
                # dataset info
                if ('data_val_info_'+str(d)) in f:
                    data_val_info = {}
                    for i in f['data_val_info_'+str(d)].keys():
                        data_val_info[i] = np.array(f['data_val_info_'+str(d)][i])
                        
                    if 'triggers' in data_val_info:
                        data_val_info['triggers'] = data_val_info['triggers'][idx_val_start:idx_val_end]
                        
                else:
                    data_val_info = None
            else:
                X = np.array(f['data_val']['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f['data_val']['y'][idx_val_start:idx_val_end],dtype='float32')
                if 'spikes' in f['data_val']:
                    spikes = np.array(f['data_val']['spikes'][idx_val_start:idx_val_end],dtype='float32')
                    data_val = Exptdata_spikes(X,y,spikes)
                else:
                    data_val = Exptdata(X,y)
    
                # dataset info
                if 'data_val_info' in f:
                    data_val_info = {}
                    for i in f['data_val_info'].keys():
                        data_val_info[i] = np.array(f['data_val_info'][i])
                else:
                    data_val_info = None
        else:
            data_val = None
    else:
        idx = np.where(bool_idx_val)
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_val = Exptdata_spikes(X,y,spikes)

       
    
    # Testing data
    if VALFROMTRAIN==False:
        if 'data_test' in f.keys():
            data_test = Exptdata(np.array(f['data_test']['X']),np.array(f['data_test']['y']))
        else:       
            data_test = None
    else:
        idx = np.where(bool_idx_test)
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_test = Exptdata_spikes(X,y,spikes)
    
    # Quality data
    select_groups = ('data_quality')
    level_keys = list(f[select_groups].keys())
    data_quality = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            data_quality[i] = h5_tostring(rgb)
        else:
            data_quality[i] = rgb
            
    # Retinal reliability data
    select_groups = ('dataset_rr')
    level_keys = list(f[select_groups].keys())
    dataset_rr = {}
    for i in level_keys:
        level4_keys = list(f[select_groups][i].keys())
        temp_2 = {}

        for d in level4_keys:
            data_key ='/'+select_groups+'/'+i+'/'+d
        
            rgb = np.array(f[data_key])
            try:
                rgb_type = rgb.dtype.name
                if 'bytes' in rgb_type:
                    temp_2[d] = h5_tostring(rgb)
                else:
                    temp_2[d] = rgb
            except:
                temp_2[d] = rgb
        dataset_rr[i] = temp_2
        
    # Parameters
    select_groups = ('parameters')
    level_keys = list(f[select_groups].keys())
    parameters = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            parameters[i] = h5_tostring(rgb)
        else:
            parameters[i] = rgb
    
    # Orig response (non normalized)
    try:
        select_groups = ('resp_orig')
        level_keys = list(f[select_groups].keys())
        resp_orig = {}
        for i in level_keys:
            data_key = '/'+select_groups+'/'+i
            rgb = np.array(f[data_key])
            rgb_type = rgb.dtype.name
               
            if 'bytes' in rgb_type:
                resp_orig[i] = h5_tostring(rgb)
            else:
                resp_orig[i] = rgb
    except:
        resp_orig = None

            
    f.close()

    if RETURN_VALINFO==False:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig,data_val_info
    
    

def model_evaluate_new(obs_rate_allStimTrials,pred_rate,filt_width,RR_ONLY=False,lag = 0,obs_noise=0):
    numCells = obs_rate_allStimTrials.shape[-1]
    if obs_rate_allStimTrials.ndim>2:
        num_trials = obs_rate_allStimTrials.shape[0]
        idx_allTrials = np.arange(num_trials)
    else:
        num_trials = 1
    
    if num_trials > 1:  # for kierstens data or where we have multiple trials of the validation data
    
        t_start = 20
        
        # obs_rate_allStimTrials_corrected = obs_rate_allStimTrials[:,filt_width:,:]
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials
        t_end = obs_rate_allStimTrials_corrected.shape[1]-t_start-20
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials_corrected[:,t_start:t_end-lag,:]
        
        # if RR_ONLY is False:
        pred_rate_corrected = pred_rate[t_start+lag:t_end,:]
        
        
        # for predicting trial averaged responses
        idx_trials_r1 = np.array(random.sample(range(0,len(idx_allTrials)),int(np.ceil(len(idx_allTrials)/2))))
        assert(np.unique(idx_trials_r1).shape[0] == idx_trials_r1.shape[0])
        idx_trials_r2 = np.setdiff1d(idx_allTrials,idx_trials_r1)
    
        r1 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r1,:,:],axis=0)
        r2 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r2,:,:],axis=0)
        
        if obs_noise==None:
            noise_trialAveraged = np.mean((r1-r2)**2,axis=0)
            
        fracExplainableVar = (np.var(r2,axis=0) - noise_trialAveraged)/(np.var(r2,axis=0)+1e-5)
        
        if RR_ONLY is True:
            fev = None
        else:
            r_pred = pred_rate_corrected
            mse_resid = np.mean((r_pred-r2)**2,axis=0)
            fev = 1 - ((mse_resid-noise_trialAveraged)/(np.var(r2,axis=0)-noise_trialAveraged))
            # fev = 1 - ((mse_resid-noise_trialAveraged)/(np.var(obs_rate_allStimTrials_corrected[idx_trials_r2,:,:],axis=(0,1))-noise_trialAveraged))
            # fev = 1 - ((mse_resid)/(np.var(r2,axis=0)-noise_trialAveraged))
        
        
        # Pearson correlation
        rr_corr = correlation_coefficient_distribution(r1,r2)
        if RR_ONLY is True:
            pred_corr = None
        else:
            pred_corr = correlation_coefficient_distribution(r2,r_pred)
            
    else:
        t_start = 10
        
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials
        t_end = obs_rate_allStimTrials_corrected.shape[0]-10
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials_corrected[t_start:t_end-lag,:]
        
        # if RR_ONLY is False:
        pred_rate_corrected = pred_rate[t_start+lag:t_end,:]

        resid = obs_rate_allStimTrials_corrected - pred_rate_corrected
        mse_resid = np.mean(resid**2,axis=0)
        var_test = np.var(obs_rate_allStimTrials_corrected,axis=0)
        fev = 1 - ((mse_resid - obs_noise)/(var_test+1e-5-obs_noise))
        fracExplainableVar = None #(var_test-obs_noise)/var_test
        
        pred_corr = correlation_coefficient_distribution(obs_rate_allStimTrials_corrected,pred_rate_corrected)
        rr_corr = None
   

    return fev, fracExplainableVar, pred_corr, rr_corr

def correlation_coefficient_distribution(obs_rate,est_rate):
    x_mu = obs_rate - np.mean(obs_rate, axis=0)
    x_std = np.std(obs_rate, axis=0)
    y_mu = est_rate - np.mean(est_rate, axis=0)
    y_std = np.std(est_rate, axis=0)
    cc_allUnits = np.mean(x_mu * y_mu,axis=0) / (x_std * y_std+1e-6)
    
    return cc_allUnits
