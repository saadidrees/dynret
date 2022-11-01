#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:53:08 2022

@author: Saad Idrees, Joel Zylberberg's Lab, York University
idrees.sa@gmail.com

This is a custom keras layer that converts light stimulus (R*/rod/s) into photoreceptor currents by using a biophysical model
of the photoreceptor by Rieke's lab https://www.biorxiv.org/content/10.1101/2021.02.13.431101v1.full

"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, MaxPool2D, BatchNormalization, GaussianNoise,LayerNormalization
from tensorflow.keras.regularizers import l1, l2


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


class photoreceptor_RODS_REIKE(tf.keras.layers.Layer):
    """
    This is a custom keras layer that converts light stimulus (R*/rod/s) into photoreceptor currents by using a biophysical model
    of the photoreceptors by Rieke's lab https://www.biorxiv.org/content/10.1101/2021.02.13.431101v1.full
    """
    def __init__(self,units=1):
        super(photoreceptor_RODS_REIKE,self).__init__()
        self.units = units
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

    def build(self,input_shape):
        sigma_init = tf.keras.initializers.Constant(0.707) # initialize to experimental values
        self.sigma = tf.Variable(name='sigma',initial_value=sigma_init(shape=(1,self.units),dtype='float32'),trainable=True)
        sigma_scaleFac = tf.keras.initializers.Constant(10.) 
        self.sigma_scaleFac = tf.Variable(name='sigma_scaleFac',initial_value=sigma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        phi_init = tf.keras.initializers.Constant(0.707)
        self.phi = tf.Variable(name='phi',initial_value=phi_init(shape=(1,self.units),dtype='float32'),trainable=True)
        phi_scaleFac = tf.keras.initializers.Constant(10.) 
        self.phi_scaleFac = tf.Variable(name='phi_scaleFac',initial_value=phi_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
       
        eta_init = tf.keras.initializers.Constant(0.0253)
        self.eta = tf.Variable(name='eta',initial_value=eta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        eta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.eta_scaleFac = tf.Variable(name='eta_scaleFac',initial_value=eta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        beta_init = tf.keras.initializers.Constant(0.25)
        self.beta = tf.Variable(name='beta',initial_value=beta_init(shape=(1,self.units),dtype='float32'),trainable=True)
        beta_scaleFac = tf.keras.initializers.Constant(100.) 
        self.beta_scaleFac = tf.Variable(name='beta_scaleFac',initial_value=beta_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        cgmp2cur_init = tf.keras.initializers.Constant(0.01)
        self.cgmp2cur = tf.Variable(name='cgmp2cur',initial_value=cgmp2cur_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cgmphill_init = tf.keras.initializers.Constant(0.3)
        self.cgmphill = tf.Variable(name='cgmphill',initial_value=cgmphill_init(shape=(1,self.units),dtype='float32'),trainable=False)
        cgmphill_scaleFac = tf.keras.initializers.Constant(10.) 
        self.cgmphill_scaleFac = tf.Variable(name='cgmphill_scaleFac',initial_value=cgmphill_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        cdark_init = tf.keras.initializers.Constant(1.)
        self.cdark = tf.Variable(name='cdark',initial_value=cdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        
        betaSlow_init = tf.keras.initializers.Constant(1.)
        self.betaSlow = tf.Variable(name='betaSlow',initial_value=betaSlow_init(shape=(1,self.units),dtype='float32'),trainable=False)
        betaSlow_scaleFac = tf.keras.initializers.Constant(1.) 
        self.betaSlow_scaleFac = tf.Variable(name='betaSlow_scaleFac',initial_value=betaSlow_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillcoef_init = tf.keras.initializers.Constant(0.4)
        self.hillcoef = tf.Variable(name='hillcoef',initial_value=hillcoef_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillcoef_scaleFac = tf.keras.initializers.Constant(10.) 
        self.hillcoef_scaleFac = tf.Variable(name='hillcoef_scaleFac',initial_value=hillcoef_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        hillaffinity_init = tf.keras.initializers.Constant(0.5)
        self.hillaffinity = tf.Variable(name='hillaffinity',initial_value=hillaffinity_init(shape=(1,self.units),dtype='float32'),trainable=False)
        hillaffinity_scaleFac = tf.keras.initializers.Constant(1.) 
        self.hillaffinity_scaleFac = tf.Variable(name='hillaffinity_scaleFac',initial_value=hillaffinity_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        gamma_init = tf.keras.initializers.Constant(1)
        self.gamma = tf.Variable(name='gamma',initial_value=gamma_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gamma_scaleFac = tf.keras.initializers.Constant(100.) 
        self.gamma_scaleFac = tf.Variable(name='gamma_scaleFac',initial_value=gamma_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)

        gdark_init = tf.keras.initializers.Constant(0.20)    # 20 for rods 
        self.gdark = tf.Variable(name='gdark',initial_value=gdark_init(shape=(1,self.units),dtype='float32'),trainable=False)
        gdark_scaleFac = tf.keras.initializers.Constant(100)    # 28 for cones; 20 for rods 
        self.gdark_scaleFac = tf.Variable(name='gdark_scaleFac',initial_value=gdark_scaleFac(shape=(1,self.units),dtype='float32'),trainable=False)
        
        self.timeBin = 8 # find a way to fix this in the model  #tf.Variable(name='timeBin',initial_value=timeBin(shape=(1,self.units),dtype='float32'),trainable=False)
 
    def call(self,inputs):
        X_fun = inputs

        timeBin = float(self.timeBin) # ms
        frameTime = 8 # ms
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

def prfr_cnn2d_rods(inputs,n_out,**kwargs):

    filt_temporal_width = kwargs['filt_temporal_width']
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    chan2_n = kwargs['chan2_n']
    filt2_size = kwargs['filt2_size']
    chan3_n = kwargs['chan3_n']
    filt3_size = kwargs['filt3_size']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    sigma = 0.1
    
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(inputs)
    y = photoreceptor_RODS_REIKE(units=1)(y)
    y = Reshape((inputs.shape[1],inputs.shape[-2],inputs.shape[-1]))(y)
    y = y[:,inputs.shape[1]-filt_temporal_width:,:,:]
    
    y = LayerNormalization(axis=[1,2,3],epsilon=1e-7)(y)
    
    # CNN - first layer
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3),name='CNNs_start')(y)
    if BatchNorm is True:
        n1 = int(y.shape[-1])
        n2 = int(y.shape[-2])
        y = Reshape((chan1_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
        
    if MaxPool is True:
        y = MaxPool2D(2,data_format='channels_first')(y)

    y = Activation('relu')(GaussianNoise(sigma)(y))
    
    # CNN - second layer
    if chan2_n>0:
        y = Conv2D(chan2_n, filt2_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan2_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))
            
        y = Activation('relu')(GaussianNoise(sigma)(y))


    # CNN - third layer
    if chan3_n>0:
        y = Conv2D(chan3_n, filt3_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
        if BatchNorm is True:
            n1 = int(y.shape[-1])
            n2 = int(y.shape[-2])
            y = Reshape((chan3_n, n2, n1))(BatchNormalization(axis=-1)(Flatten()(y)))

        y = Activation('relu')(GaussianNoise(sigma)(y))

    
    # Dense layer
    y = Flatten()(y)
    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus',dtype='float32')(y)

    mdl_name = 'PRFR_CNN2D_RODS'
    return Model(inputs, outputs, name=mdl_name)

