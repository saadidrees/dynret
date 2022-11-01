#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

"""

import numpy as np
import tensorflow as tf
import model_pr.prmodel
from tensorflow.keras.layers import Input


# GENERATE RANDOM DATA
"""
Input stimulus should be in units of R*/rod/s
X should be the stimulus input to the model with dims [samples,frames,pixel,pixel]
y should be the response of RGCs at frame[-1] for each sample in X. Dims are [samples,N_rgcs]
For a continous time-varying stimulus that is rolled, samples will represent the time dimension.
Samp1 will contain frames 0-100
Samp2 will contain frames 1-101
Samp3 will contain frames 2-102
and so on
"""
n_rgcs = 40             # number of output units/cells
nsamples = 10000         # number of input samples to generate
X = np.random.rand(nsamples,39,30)           # stimulus [nsamples(this is also the time axis),pixels,pixels]
y = np.random.rand(nsamples,n_rgcs)      # response of n_rgcs neurons - one response value for each sample

# Roll stimulus along the time dimension to generate a temporal window for each sample
width_temporal = 180        # this is the stimulus history on which the photoreceptor model will operate on
width_temporal_final = 120     # The photoreceptor output will be trimmed to maintain a history of these many samples (to get rid of boundary effects)

X = X.T
shape = X.shape[:-1] + (X.shape[-1] - width_temporal, width_temporal)
strides = X.strides + (X.strides[-1],)
X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

X = np.rollaxis(X.T, 1, 0)  # now X has an additional dimension [samp,samp_history,pixels,pixels]
y = y[width_temporal:]      # trim y response according to the temporal width


# %% BUILD THE MODEL - PHOTORECEPTOR --> 3xCNN --> DENSE

pr_temporal_width = width_temporal
temporal_width=width_temporal_final
chan1_n=8
filt1_size=9
chan2_n=16
filt2_size=7
chan3_n=18
filt3_size=5
bz=125
BatchNorm=1
MaxPool=1

dict_params = dict(chan1_n=chan1_n,filt1_size=filt1_size,
                   chan2_n=chan2_n,filt2_size=filt2_size,
                   chan3_n=chan3_n,filt3_size=filt3_size,
                   filt_temporal_width=temporal_width,
                   BatchNorm=BatchNorm,MaxPool=MaxPool)

inp_shape = Input(shape=X.shape[1:]) # keras input layer
mdl = model_pr.prmodel.prfr_cnn2d_rods(inp_shape,n_rgcs,**dict_params)
mdl.summary()

# %% Train model
lr = 0.0001
nb_epochs=10

mdl.compile(loss='poisson', optimizer=tf.keras.optimizers.Adam(lr))
mdl_history = mdl.fit(x=X, y=y, batch_size=125, epochs=nb_epochs)



