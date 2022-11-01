#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01 Nov 2022
@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

This script demonstrates how to use the adaptive-conv layer.
Cell 1 generates training and validation data
Cell 2 trains a model with adaptive-conv as the input layer
Cell 3 trains a model with standard convolutions (no adaptive-conv)
"""

import model_acnn.acnn, model_acnn.train_model, model_acnn.stimuli
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from model_acnn.utils import rolling_window
import matplotlib.pyplot as plt
from collections import namedtuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])


# %% 1. Generate data   
"""
- Input to the model, stimulus, is the intensity of one pixel varying in time.
- This intensity is composed by multiplying 2 signals:
    1. Itensity of an object varying across a mean value with small amplitude
    2. Intensity of a source that is mostly at its mean value but at random times
    and for random durations this intensity changes by log units.
- Target or the model's output is the intensity of the object, discounting the large source variations
- The intensity signals are governed by three parameter:
    1. Frequency of the change given by timeBin variable
    2. mean intensity
    3. amplitude
- These values are taken in by the function stimuli.obj_source_multi which outputs the two signals.
- The resulting signals are 1D. representing one pixel's value in time. 
- The model has convolutions operating in time. So we roll the 1D signal resulting in 2D where the first
dimension is the sample and the second dimension is frames of length temporal_width. So the first sample
contains 1-200 frames of the original 1D signal. Second sample contains 2-201, third sample: 3-202 and so on
- A portion of these samples are kept for the validation set
- Signleton dimensions are added to the data for representing the x-y pixel.
- data_train.X is the input and data_train.y is the target
"""
totalTime = 10 #mins - length of training data
temporal_width=200  # This will be size of temporal kernel and also number of frames (time) in one training

# object intensity parameters
timeBin_obj = 10    # underlies frequency
mean_obj = 5        # mean intensity level
amp_obj = 1         # intensity amplitude

# source intensity parameters
mean_src = 5
timeBin_src = 500
dur_src = np.array([40,60,80]) 
amp_src = np.array([1e1,1e2,1e3,1e4,1e5]) # large intensities

# get the intensity signals using above params
lum_obj,lum_src = model_acnn.stimuli.obj_source_multi(totalTime=totalTime,timeBin_obj=timeBin_obj,mean_obj=mean_obj,amp_obj=amp_obj,
                                                      timeBin_src = timeBin_src,mean_src=mean_src,amp_src=amp_src,dur_src=dur_src,
                                                      sigma=1,temporal_width=temporal_width,frac_perturb=1)

# Model input (stim) is the combined intensity of obj and src. Model should extract the obj intensity (resp).
stim = lum_obj*lum_src
resp = lum_obj

# Visualize the stimulus
idx_plots = np.arange(500,1500)
plt.plot(lum_src[idx_plots])
plt.plot(lum_obj[idx_plots],'r')
plt.show()
plt.plot(stim[idx_plots],'g',linewidth=3)
plt.plot(mean_src*lum_obj[idx_plots],'r')        
plt.show()


# ---- Training and validation datasets
# Training set
frac_train = 0.9    # remaining is used for validation
idx_train = np.floor(frac_train*stim.shape[0]).astype('int')

stim_train = stim[:idx_train].copy()
spike_vec_train = resp[:idx_train].copy() 

stim_test = stim[idx_train:].copy()
spike_vec_test = resp[idx_train:].copy()

N_test_limit = 20000
if stim_test.shape[0]>N_test_limit:
    stim_test = stim_test[:N_test_limit]
    spike_vec_test = spike_vec_test[:N_test_limit]

dict_train = dict(
    X=stim_train,
    y = spike_vec_train,
    )
del stim_train, spike_vec_train

dict_val = dict(
    X=stim_test,
    y = spike_vec_test,
    )
del stim_test, spike_vec_test

X = dict_train['X'] # 1 pixel intensity varying in time
X = rolling_window(X,temporal_width)    # Make samples from above where each sample is of frames [temporal_width].
X = X[:,:,np.newaxis,np.newaxis]        # Add singleton dimensions for 1x1 pixel
y = dict_train['y']                     # 
y = rolling_window(y,temporal_width)    # Do the same with model output
y = y[:,-1]                             # The model output is the last frame of each sample. So the model basically needs to infer the intensity of last frame (i.e. current time point) from previous 200 (temporal_width) frames

if y.ndim==1:                           # Add a singleton dimension for number of output units
    y = y[:,np.newaxis]
data_train = Exptdata(X,y)

# Validation set
X = dict_val['X']
X = rolling_window(X,temporal_width)
X = X[:,:,np.newaxis,np.newaxis]

y = dict_val['y']
y = rolling_window(y,temporal_width)
y_val_rolled = y.copy()
y = y[:,-1] #y = y[:,-1]
if y.ndim==1:
    y = y[:,np.newaxis]
data_val = Exptdata(X,y)

del X, y


inputs = Input(data_train.X.shape[1:]) # keras input layer
n_out = data_train.y.shape[1]         # number of units in output layer


# %% 2. Adaptive-Conv

chan1_n=40      # Number of units in the adaptive-conv input layer
filt1_size=1    # spatial filter size
N_layers = 7    # Number of dense layers following adaptive-conv
chan2_n=80      # number of units in each dense layer

MaxPool=0
BatchNorm = 1

dict_params = dict(filt_temporal_width=temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,N_layers=N_layers,chan2_n=chan2_n,
                   BatchNorm=BatchNorm,MaxPool=MaxPool,)
   
mdl = model_acnn.acnn.A_CNN_DENSE(inputs, n_out, **dict_params)      # builds the model with adaptive-conv layer
mdl.summary()

lr = 0.01
use_lrscheduler = 1
mdl_history = model_acnn.train_model.train(mdl, data_train, data_val, bz=4096, nb_epochs=100,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=1)  


optimizer = Adam(lr)
mdl.compile(loss='poisson', optimizer=optimizer)

mdl_history = mdl.fit(x=data_train.X, y=data_train.y, batch_size=4096, epochs=100,validation_data=(data_val.X,data_val.y),
                       shuffle=True,use_multiprocessing=True)

y_orig = data_val.y
y_adaptiveConv = mdl.predict(data_val.X)
plt.plot(y_orig[1000:2000]);plt.plot(y_adaptiveConv[1000:2000]);plt.show()


# %% 3. Standard CNNs

chan1_n=100         # Number of units of the standard conv input layer
filt1_size=1        # spatial filter size
N_layers = 5        # Number of dense layers following the standard conv layer
chan2_n=200         # number of units in each dense layer

MaxPool=0
BatchNorm = 1

dict_params = dict(filt_temporal_width=temporal_width,chan1_n=chan1_n,filt1_size=filt1_size,N_layers=N_layers,chan2_n=chan2_n,
                   BatchNorm=BatchNorm,MaxPool=MaxPool,)
    
mdl = model_acnn.acnn.CNN_DENSE(inputs, n_out, **dict_params)      # builds the model with standard-conv layer
mdl.summary()

lr = 0.01
use_lrscheduler = 1

mdl_history = model_acnn.train_model.train(mdl, data_train, data_val, bz=4096, nb_epochs=100,lr=lr,use_lrscheduler=use_lrscheduler,lr_fac=1)  

y_standardConv = mdl.predict(data_val.X)
plt.plot(y_orig[1000:5000]);plt.plot(y_standardConv[1000:5000]);plt.show()

