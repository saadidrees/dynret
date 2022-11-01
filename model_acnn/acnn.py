#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:42:39 2021

@author: saad
""" 

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, Reshape, MaxPool2D, Permute, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.regularizers import l1, l2
import numpy as np
import math



# %% Adaptive-Conv

def generate_simple_filter_multichan(tau,n,t):

    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    f = (t**n[:,None])*tf.math.exp(-t/tau[:,None]) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb[:,None])/tf.math.exp(tf.math.lgamma(n+1))[:,None] # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)
   
    return f

""" test filter 

    tau = tf.constant([[1]],dtype=tf.float32)
    n = tf.constant([[1]],dtype=tf.float32)
    t = tf.range(0,1000/timeBin,dtype='float32')
    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    tN = np.squeeze(t.eval(session=tf.compat.v1.Session()))
    a = t**n; aN = np.squeeze(a.eval(session=tf.compat.v1.Session()))
    f = (t**n)*tf.math.exp(-t/tau) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb)/tf.math.exp(tf.math.lgamma(n+1)) # normalize appropriately
    
    f = np.squeeze(f.eval(session=tf.compat.v1.Session()))
    plt.plot(f)


"""


def conv_oper_multichan(x,kernel_1D):
    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D,axis=0)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims,1])
    kernel_reshaped = tf.tile(kernel_1D,(tile_fac))
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,kernel_1D.shape[0],kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-2,0)
    pad_vec = [[0,0],[kernel_1D.shape[0]-1,0],[0,0],[0,0]]
    # pad_vec = [[0,0],[0,0],[0,0],[0,0]]
    conv_output = tf.nn.conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    # print(conv_output.shape)
    return conv_output

@tf.function
def slice_tensor(inp_tensor,shift_vals):
    # print(inp_tensor.shape)
    # print(shift_vals.shape)
    tens_reshape = tf.reshape(inp_tensor,[-1,inp_tensor.shape[1]*inp_tensor.shape[2]*inp_tensor.shape[3]*inp_tensor.shape[4]])
    shift_vals_new = ((inp_tensor.shape[1]-shift_vals[0,:])*shift_vals.shape[-1]) + tf.range(0,shift_vals.shape[-1])
    extracted_vals = tf.gather(tens_reshape,shift_vals_new,axis=1)
    extracted_vals_reshaped = tf.reshape(extracted_vals,(-1,1,inp_tensor.shape[2],inp_tensor.shape[3],inp_tensor.shape[4]))
    
    return extracted_vals_reshaped
    

# Adaptive-Conv custom layer
class photoreceptor_DA_multichan_randinit(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_randinit,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
            
    
    # Initial conditions matter! Need a better way to fix this.
    def build(self,input_shape):    # random inits
    
        zeta_range = (0.00,0.01)
        zeta_init = tf.keras.initializers.RandomUniform(minval=zeta_range[0],maxval=zeta_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,zeta_range[0],zeta_range[1]))
        zeta_mulFac = tf.keras.initializers.Constant(1000.) 
        self.zeta_mulFac = self.add_weight(name='zeta_mulFac',initializer=zeta_mulFac,shape=[1,self.units],trainable=False)
        
        kappa_range = (0.00,0.01)
        kappa_init = tf.keras.initializers.RandomUniform(minval=kappa_range[0],maxval=kappa_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.kappa = self.add_weight(name='kappa',initializer=kappa_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,kappa_range[0],kappa_range[1]))
        kappa_mulFac = tf.keras.initializers.Constant(1000.) 
        self.kappa_mulFac = self.add_weight(name='kappa_mulFac',initializer=kappa_mulFac,shape=[1,self.units],trainable=False)
        
        alpha_range = (0.001,0.1)
        alpha_init = tf.keras.initializers.RandomUniform(minval=alpha_range[0],maxval=alpha_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(100.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.001,0.1)
        beta_init = tf.keras.initializers.RandomUniform(minval=beta_range[0],maxval=beta_range[1])  #tf.keras.initializers.Constant(0.02)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)

        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.RandomUniform(minval=gamma_range[0],maxval=gamma_range[1])  #tf.keras.initializers.Constant(0.075)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.001,0.02)
        tauY_init = tf.keras.initializers.RandomUniform(minval=tauY_range[0],maxval=tauY_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(100.) #tf.keras.initializers.Constant(10.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.1)
        nY_init = tf.keras.initializers.RandomUniform(minval=nY_range[0],maxval=nY_range[1]) #tf.keras.initializers.Constant(0.01)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


        tauZ_range = (0.3,1.)
        tauZ_init = tf.keras.initializers.RandomUniform(minval=tauZ_range[0],maxval=tauZ_range[1]) #tf.keras.initializers.Constant(0.5)# 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.01) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        tauC_range = (0.01,0.5)
        tauC_init = tf.keras.initializers.RandomUniform(minval=tauC_range[0],maxval=tauC_range[1])  #tf.keras.initializers.Constant(0.2)# 
        self.tauC = self.add_weight(name='tauC',initializer=tauC_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauC_range[0],tauC_range[1]))
        tauC_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauC_mulFac = tf.Variable(name='tauC_mulFac',initial_value=tauC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nC_range = (1e-5,0.5)
        nC_init = tf.keras.initializers.Constant(0.01) # tf.keras.initializers.RandomUniform(minval=nC_range[0],maxval=nC_range[1]) # 
        self.nC = self.add_weight(name='nC',initializer=nC_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nC_range[0],nC_range[1]))
        nC_mulFac = tf.keras.initializers.Constant(10.) 
        self.nC_mulFac = tf.Variable(name='nC_mulFac',initial_value=nC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha*self.alpha_mulFac
        beta = self.beta*self.beta_mulFac
        gamma =  self.gamma*self.gamma_mulFac
        zeta = self.zeta*self.zeta_mulFac
        kappa = self.kappa*self.kappa_mulFac
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        tau_c =  (self.tauC_mulFac*self.tauC) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        n_c =  (self.nC_mulFac*self.nC)
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        # print(tau_y.shape)
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kc = generate_simple_filter_multichan(tau_c,n_c,t)  
        Kz = generate_simple_filter_multichan(tau_z,n_z,t)  
        Kz = (gamma*Kc) + ((1-gamma) * Kz)
        
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
        # print('z_tf:'+z_tf.shape)
               
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        # print('z_tf shape:'+z_tf_reshape.shape)
        
        y_shift = tf.math.argmax(Ky,axis=1);y_shift = tf.cast(y_shift,tf.int32)
        z_shift = tf.math.argmax(Kz,axis=1);z_shift = tf.cast(z_shift,tf.int32)
        
        y_tf_reshape = slice_tensor(y_tf_reshape,y_shift)
        z_tf_reshape = slice_tensor(z_tf_reshape,z_shift)
        # print('z_tf shape_sliced:'+z_tf_reshape.shape)
               
    
        outputs = (zeta[None,None,0,None,:] + (alpha[None,None,0,None,:]*y_tf_reshape[:,:,0,:,:]))/(kappa[None,None,0,None,:]+1e-6+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))
        
        return outputs



def A_CNN_DENSE(inputs,n_out,**kwargs): # A-conv --> X Dense --> Dense
    
    chan1_n = kwargs['chan1_n']; filt1_size = kwargs['filt1_size']
    N_layers = kwargs['N_layers']
    chan2_n = kwargs['chan2_n']; 
    BatchNorm = bool(kwargs['BatchNorm']); MaxPool = bool(kwargs['MaxPool'])
    
    y = inputs
    y = Reshape((inputs.shape[1],inputs.shape[-2]*inputs.shape[-1]))(y)
    
    # adaptive-conv layer
    y = photoreceptor_DA_multichan_randinit(units=chan1_n,kernel_regularizer=l2(1e-4))(y)
    y = Reshape((1,inputs.shape[-2],inputs.shape[-1],chan1_n))(y)
    y = Permute((4,2,3,1))(y)   # Channels first
    y = y[:,:,:,:,0]       # only take the first time point
    
    # dense layers
    if chan1_n==1 and chan2_n<1:
        y = Activation('softplus')(y)
    else:
        y = Activation('relu')(y)
        
        if N_layers>0 and chan2_n>0:
            y = Flatten()(y)
            
            for i in range(N_layers):
                if N_layers>2:
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)
                else:
                    y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-4))(y)

                if BatchNorm is True: 
                    rgb = y.shape[1:]
                    y = Reshape(rgb)(BatchNormalization(axis=-1)(Flatten()(y)))
                y = Activation('relu')(y)

        
    # Output layer
    y = Flatten()(y)

    if BatchNorm is True: 
        y = BatchNormalization(axis=-1)(y)
    y = Dense(n_out, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)    

    mdl_name = 'A_CNN_DENSE'
    return Model(inputs, outputs, name=mdl_name)



# %% Conventional CNN model

def CNN_DENSE(inputs,n_out,**kwargs): # standard-conv --> X Dense --> Dense
    
    chan1_n = kwargs['chan1_n']
    filt1_size = kwargs['filt1_size']
    N_layers = kwargs['N_layers']
    chan2_n = kwargs['chan2_n']
    BatchNorm = bool(kwargs['BatchNorm'])
    MaxPool = bool(kwargs['MaxPool'])
    
    y = inputs
    
    # Standard Conv layer 
    y = Conv2D(chan1_n, filt1_size, data_format="channels_first", kernel_regularizer=l2(1e-3))(y)
    if BatchNorm is True:
        rgb = y.shape
        y = Flatten()(y)
        y = BatchNormalization()(y)
        y = Reshape((rgb[1],rgb[2],rgb[3]))(y)
        
    y = Activation('relu')(y)
    
    if chan1_n==1 and chan2_n<1:
        outputs = y
        
    else:
    # Dense layers
        if N_layers>0:
            y = Flatten()(y)

            for i in range(N_layers):
                y = Dense(chan2_n, kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)

                if BatchNorm is True:
                    y = BatchNormalization()(y)
                y = Activation('relu')(y)
                          
    # Output layer
    if BatchNorm is True:
        y = BatchNormalization()(y)
    y = Dense(n_out, kernel_initializer='normal', kernel_regularizer=l2(1e-3), activity_regularizer=l1(1e-3))(y)
    outputs = Activation('softplus')(y)

    mdl_name = 'CNN_DENSE'
    return Model(inputs, outputs, name=mdl_name)
