#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

"""

# import needed modules
import numpy as np
import os

import models
from data_handler import prepare_data_cnn2d, load_h5Dataset, model_evaluate_new


from tensorflow.keras.layers import Input


import gc
import tensorflow as tf
print(tf.__version__)
tf.compat.v1.disable_eager_execution()


    
# load train val and test datasets from saved h5 file
"""
    load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    and data_train.y contains the spikerate normalized by median [samples,numOfCells]
"""


fname_data_train_val_test_all = '/home/saad/postdoc_db/analyses/data_kiersten/monkey01/datasets/monkey01_dataset_train_val_test_scot-30-Rstar.h5'


idx_train_start = 0    # mins to chop off in the begining.
trainingSamps_dur = 30      # minutes (total is like 50 mins i think. If you put -1 here then you will load all the data
validationSamps_dur = 1 # minutes
testSamps_dur = 0.3 # minutes

rgb = load_h5Dataset(fname_data_train_val_test_all,nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                     idx_train_start=idx_train_start)
data_train=rgb[0]
data_val = rgb[1]
data_test = rgb[2]
data_quality = rgb[3]
dataset_rr = rgb[4]
parameters = rgb[5]
if len(rgb)>7:
    data_info = rgb[7]

t_frame = parameters['t_frame']     # time in ms of one frame/sample 

"""
data_train.X contains the movie frames [time,y,x]
data_train.y contains the firing rate [time,rgc]

Same for data_val and data_test. The three datasets (train, val and test) will not overlap

"""

# %% Arrange data in training samples format. Roll the time dimension to have movie chunks equal to the temporal width of your CNN
temporal_width = 120   # frames (120 frames ~ 1s)
idx_unitsToTake = np.arange(0,data_train.y.shape[-1])   # Take all the rgcs
data_train = prepare_data_cnn2d(data_train,temporal_width,idx_unitsToTake)     # [samples,temporal_width,rows,columns]
data_test = prepare_data_cnn2d(data_test,temporal_width,idx_unitsToTake)
data_val = prepare_data_cnn2d(data_val,temporal_width,idx_unitsToTake)   


# %% Build a conventional-CNN

chan1_n=10
filt1_size=11
chan2_n=15
filt2_size=7
chan3_n=20
filt3_size=7
bz=125
BatchNorm=1
MaxPool=1


dict_params = dict(chan1_n=chan1_n,filt1_size=filt1_size,
                   chan2_n=chan2_n,filt2_size=filt2_size,
                   chan3_n=chan3_n,filt3_size=filt3_size,
                   filt_temporal_width=temporal_width,
                   BatchNorm=BatchNorm,MaxPool=MaxPool,
                   dtype='float32')

n_rgcs = data_train.y.shape[-1]        # number of units in output layer
inp_shape = Input(shape=data_train.X.shape[1:]) # keras input layer

mdl = models.cnn2d(inp_shape,n_rgcs,**dict_params)
mdl.summary()

# %% Train model
lr = 0.0001
nb_epochs=100

mdl.compile(loss='poisson', optimizer=tf.keras.optimizers.legacy.Adam(lr))
mdl_history = mdl.fit(x=data_train.X, y=data_train.y, validation_data=(data_val.X,data_val.y), validation_freq=1, shuffle=True,batch_size=bz, epochs=nb_epochs)


# %% Model Evaluation
fev_allUnits = np.zeros((nb_epochs,n_rgcs))
fev_allUnits[:] = np.nan

predCorr_allUnits = np.zeros((nb_epochs,n_rgcs))
predCorr_allUnits[:] = np.nan

# for compatibility with greg's dataset


obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
obs_noise = None
num_iters = 10


samps_shift = 0 


pred_rate = mdl.predict(data_test.X)

fev_loop = np.zeros((num_iters,n_rgcs))
fracExVar_loop = np.zeros((num_iters,n_rgcs))
predCorr_loop = np.zeros((num_iters,n_rgcs))
rrCorr_loop = np.zeros((num_iters,n_rgcs))

for j in range(num_iters):  # nunm_iters is 1 with my dataset. This was mainly for greg's data where we would randomly split the dataset to calculate performance metrics 
    fev_loop[j,:], fracExVar_loop[j,:], predCorr_loop[j,:], rrCorr_loop[j,:] = model_evaluate_new(obs_rate_allStimTrials,pred_rate,temporal_width,lag=int(samps_shift),obs_noise=obs_noise)
    
fev = np.mean(fev_loop,axis=0)


fev_allUnits = np.mean(fev_loop,axis=0)
fev_medianUnits = np.nanmedian(fev_allUnits)      

predCorr_allUnits = np.mean(predCorr_loop,axis=0)
predCorr_medianUnits = np.nanmedian(predCorr_allUnits)

_ = gc.collect()



# %% Photoreceptor-CNN model

"""

pr_temporal_width = width_temporal
temporal_width=width_temporal_final
chan1_n=10
filt1_size=15
chan2_n=15
filt2_size=11
chan3_n=25
filt3_size=11
bz=125
BatchNorm=1
MaxPool=1

pr_params = prfr_params.fr_rods_trainable()        # fr_rods_trainable, fr_rods_fixed, fr_cones_trainable, fr_cones_fixed

dict_params = dict(chan1_n=chan1_n,filt1_size=filt1_size,
                   chan2_n=chan2_n,filt2_size=filt2_size,
                   chan3_n=chan3_n,filt3_size=filt3_size,
                   filt_temporal_width=temporal_width,
                   BatchNorm=BatchNorm,MaxPool=MaxPool,
                   pr_params=pr_params,
                   dtype='float32')


inp_shape = Input(shape=X.shape[1:]) # keras input layer
mdl = models.prfr_cnn2d(inp_shape,n_rgcs,**dict_params)
mdl.summary()

# % Train model
lr = 0.0001
nb_epochs=10

mdl.compile(loss='poisson', optimizer=tf.keras.optimizers.Adam(lr))
mdl_history = mdl.fit(x=X, y=y, batch_size=125, epochs=nb_epochs)




"""
