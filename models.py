#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:53:08 2022

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

This is a custom keras layer that converts light stimulus (R*/rod/s) into photoreceptor currents by using a biophysical model
of the photoreceptor by Rieke's lab "Predicting and Manipulating Cone Responses to Naturalistic Inputs. Juan M. Angueyra, Jacob Baudin, Gregory W. Schwartz, Fred Rieke
Journal of Neuroscience 16 February 2022, 42 (7) 1254-1274; DOI: 10.1523/JNEUROSCI.0793-21.2021


"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, MaxPool2D, BatchNormalization, GaussianNoise,LayerNormalization
from tensorflow.keras.regularizers import l1, l2


# Photoreceptor-CNN model
# Solve the differential equations using Euler method
@tf.function(autograph=True,experimental_relax_shapes=True)
def riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark):
    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    tme = tf.range(0,X_fun.shape[1],dtype='float32')*TimeStep
    NumPts = tme.shape[0]
    
# initial conditions   
    g_prev = gdark+(X_fun[:,0,:]*0)
    s_prev = (gdark * eta/phi)+(X_fun[:,0,:]*0)
    c_prev = cdark+(X_fun[:,0,:]*0)
    r_prev = X_fun[:,0,:] * gamma / sigma
    p_prev = (eta + r_prev)/phi

    g = tf.TensorArray(tf.float32,size=NumPts)
    g.write(0,X_fun[:,0,:]*0)
    
    # solve difference equations
    for pnt in tf.range(1,NumPts):
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:,pnt-1,:]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * (cgmp2cur * g_prev **cgmphill)/2 - beta * c_prev)
        s_curr = smax / (1 + (c_curr / hillaffinity) **hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.write(pnt,g_curr)
        
        
        # update prev values to current
        g_prev = g_curr
        s_prev = s_curr
        c_prev = c_curr
        p_prev = p_curr
        r_prev = r_curr
    
    g = g.stack()
    g = tf.transpose(g,(1,0,2))
    outputs = -(cgmp2cur * g **cgmphill)/2
    
    return outputs


class photoreceptor_REIKE(tf.keras.layers.Layer):
    def __init__(self,pr_params,units=1,dtype='foat16'):
        super(photoreceptor_REIKE,self).__init__()
        self.units = units
        self.pr_params = pr_params
        # self.dtype='float16'
        

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        dtype = self.dtype
        
        sigma_init = tf.keras.initializers.Constant(self.pr_params['sigma'])
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['sigma_trainable'])
        sigma_scaleFac = tf.keras.initializers.Constant(self.pr_params['sigma_scaleFac'])
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(self.pr_params['phi'])
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['phi_trainable'])
        phi_scaleFac = tf.keras.initializers.Constant(self.pr_params['phi_scaleFac'])
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(self.pr_params['eta'])
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['eta_trainable'])
        eta_scaleFac = tf.keras.initializers.Constant(self.pr_params['eta_scaleFac'])
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(self.pr_params['beta'])
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['beta_trainable'])
        beta_scaleFac = tf.keras.initializers.Constant(self.pr_params['beta_scaleFac'])
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(self.pr_params['cgmp2cur'])
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cgmp2cur_trainable'])
        
        cgmphill_init = tf.keras.initializers.Constant(self.pr_params['cgmphill'])
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cgmphill_trainable'])
        cgmphill_scaleFac = tf.keras.initializers.Constant(self.pr_params['cgmphill_scaleFac'])
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(self.pr_params['cdark'])
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['cdark_trainable'])
        
        betaSlow_init = tf.keras.initializers.Constant(self.pr_params['betaSlow'])
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['betaSlow_trainable'])
        betaSlow_scaleFac = tf.keras.initializers.Constant(self.pr_params['betaSlow_scaleFac'])
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(self.pr_params['hillcoef'])
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['hillcoef_trainable'])
        hillcoef_scaleFac = tf.keras.initializers.Constant(self.pr_params['hillcoef_scaleFac'])
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(self.pr_params['hillaffinity'])
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['hillaffinity_trainable'])
        hillaffinity_scaleFac = tf.keras.initializers.Constant(self.pr_params['hillaffinity_scaleFac'])
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(self.pr_params['gamma'])
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['gamma_trainable'])
        gamma_scaleFac = tf.keras.initializers.Constant(self.pr_params['gamma_scaleFac'])
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)
                
        gdark_init = tf.keras.initializers.Constant(self.pr_params['gdark'])
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype=dtype),trainable=self.pr_params['gdark_trainable'])
        gdark_scaleFac = tf.keras.initializers.Constant(self.pr_params['gdark_scaleFac'])
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype=dtype),trainable=False)


 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.pr_params['timeBin']) # ms
        frameTime = timeBin # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3*timeBin
        
        if upSamp_fac>1:
            X_fun = tf.keras.backend.repeat_elements(X_fun,upSamp_fac,axis=1) 
            X_fun = X_fun/upSamp_fac     # appropriate scaling for photons/ms

        sigma = self.sigma * self.sigma_scaleFac
        phi = self.phi * self.phi_scaleFac
        eta = self.eta * self.eta_scaleFac
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.cgmphill_scaleFac
        cdark = self.cdark
        beta = self.beta * self.beta_scaleFac
        betaSlow = self.betaSlow * self.betaSlow_scaleFac
        hillcoef = self.hillcoef * self.hillcoef_scaleFac
        hillaffinity = self.hillaffinity * self.hillaffinity_scaleFac
        gamma = (self.gamma*self.gamma_scaleFac)/timeBin
        gdark = self.gdark*self.gdark_scaleFac
        
        
        outputs = riekeModel(X_fun,TimeStep,sigma,phi,eta,cgmp2cur,cgmphill,cdark,beta,betaSlow,hillcoef,hillaffinity,gamma,gdark)
        
        if upSamp_fac>1:
            outputs = outputs[:,upSamp_fac-1::upSamp_fac]
        return outputs

def prfr_cnn2d(inputs,n_out,**kwargs):

    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    filt_temporal_width=kwargs['filt_temporal_width']

    pr_params = kwargs['pr_params']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    
    y = inputs
    # y = BatchNormalization(axis=-3,epsilon=1e-7)(y)
    y = Reshape((y.shape[1],y.shape[-2]*y.shape[-1]),dtype=dtype)(y)
    y = photoreceptor_REIKE(pr_params,units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    y = LayerNormalization(axis=-3,epsilon=1e-7)(y)      # Along the temporal axis

    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
        
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D'
    return Model(inputs, outputs, name=mdl_name)


# %% Conventional CNN model
def cnn2d(inputs,n_out,**kwargs): #(inputs, n_out, chan1_n=12, filt1_size=13, chan2_n=0, filt2_size=0, chan3_n=0, filt3_size=0, BatchNorm=True, BatchNorm_train=False, MaxPool=False):
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = kwargs['MaxPool']
    dtype = kwargs['dtype']
    
    mdl_params = {}
    keys = ('chan4_n','filt4_size')
    for k in keys:
        if k in kwargs:
            mdl_params[k] = kwargs[k]
        else:
            mdl_params[k] = 0
    
    sigma = 0.1
    filt_temporal_width=inputs.shape[1]

    # first layer  
    y = inputs
    y = LayerNormalization(axis=-3,epsilon=1e-7,trainable=False)(y)        # z-score the input across temporal dimension
    # y = LayerNormalization(epsilon=1e-7)(y)        # z-score the input
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    
    if MaxPool > 0:
        if MaxPool==1:  # backwards compatibility
            MaxPool=2
        y = MaxPool2D(MaxPool,data_format='channels_first')(y)
        
    if BatchNorm is True: 
        y = BatchNormalization(axis=1,epsilon=1e-7)(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))


    # second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)                  
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)
            
        y = Activation('relu')(GaussianNoise(sigma)(y))

    # Third layer
    if chan3_n>0:
        if y.shape[-1]<filt3_size:
            filt3_size = (filt3_size,y.shape[-1])
        elif y.shape[-2]<filt3_size:
            filt3_size = (y.shape[-2],filt3_size)
        else:
            filt3_size = filt3_size
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # Fourth layer
    if mdl_params['chan4_n']>0:
        if y.shape[-1]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (mdl_params['filt4_size'],y.shape[-1])
        elif y.shape[-2]<mdl_params['filt4_size']:
            mdl_params['filt4_size'] = (y.shape[-2],mdl_params['filt4_size'])
        else:
            mdl_params['filt4_size'] = mdl_params['filt4_size']
            
        y = Conv2D(mdl_params['chan4_n'], mdl_params['filt4_size'], data_format="channels_first", kernel_regularizer=l2(1e-3))(y)    
        
        if BatchNorm is True: 
            y = BatchNormalization(axis=1,epsilon=1e-7)(y)

        y = Activation('relu')(GaussianNoise(sigma)(y))

        
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)
    # outputs = Activation('relu')(y)

    mdl_name = 'CNN2D'
    return Model(inputs, outputs, name=mdl_name)