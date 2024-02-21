#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:01:57 2022

@author: saad
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d

def obj_source(totalTime,timeBin_obj=10,mean_obj=10,amp_obj=2,timeBin_src = 50,mean_src=10,amp_src=5,sigma=1):
        
        lum_obj = np.random.normal(mean_obj,amp_obj,int(totalTime*60*1000/timeBin_obj))
        lum_obj = np.repeat(lum_obj,timeBin_obj)
        lum_obj = gaussian_filter(lum_obj,sigma=sigma)
        
        numpts = 1e-3*lum_obj.shape[0]
        t = np.arange(0,numpts,1e-3)
        f = 1000/timeBin_src
        w = 2*np.pi*f # rad/s
        lum_src = mean_src + (amp_src*np.sin(w*t+np.pi))
        
        # lum_src = np.random.normal(mean_src,amp_src,int(totalTime*60*1000/timeBin_src))
        # lum_src = np.repeat(lum_src,timeBin_src)
        # lum_src = gaussian_filter(lum_src,sigma=sigma)
        
        
        stim = lum_src*lum_obj
        resp = lum_obj.copy()
        
        return stim,resp,lum_src,lum_obj


def sin_mul(totalTime,freq_obj=5,amp_obj=1,offset_obj=1,freq_src=1,amp_src=1,offset_src=1):
    t = np.arange(0,totalTime,1e-3)

    w = 2*np.pi*freq_obj # rad/s
    lum_obj = offset_obj + (amp_obj*np.sin(w*t+np.pi))
    
    w = 2*np.pi*freq_src # rad/s
    lum_src = offset_src + (amp_src*np.sin(w*t+np.pi))
    

    stim = lum_src*lum_obj
    resp = lum_obj.copy()
    
    return stim,resp,lum_src,lum_obj

def obj_source_multi(totalTime,timeBin_obj=10,mean_obj=10,amp_obj=2,timeBin_src = 50,mean_src=10,amp_src=5,dur_src=50,sigma=1,temporal_width=0,frac_perturb=1):
    
    nsamps_obj =  int(totalTime*60*1000/timeBin_obj)+timeBin_src
    lum_obj = np.random.normal(mean_obj,amp_obj,nsamps_obj)
    lum_obj = np.repeat(lum_obj,timeBin_obj)
    lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    
    lum_obj_temp = np.reshape(lum_obj,(int(lum_obj.shape[0]/timeBin_src),int(timeBin_src)),order='C')
    lum_obj_rever = np.reshape(lum_obj_temp,lum_obj.shape[0],order='C')
    assert np.sum(abs(lum_obj-lum_obj_rever))==0
    
    range_shift = np.arange((2*timeBin_obj),timeBin_src-dur_src.max()-(2*timeBin_obj),timeBin_obj,dtype='int32')
 
    total_conds = dur_src.size * amp_src.size
    step_block = np.zeros((timeBin_src,total_conds))
    dur_amp_shift_id = np.zeros((total_conds,2))
    cntr = -1
    for i in range(dur_src.size):
        for j in range(amp_src.size):
            # for k in range_shift:
            cntr+=1
            dur_amp_shift_id[cntr,0] = dur_src[i]
            dur_amp_shift_id[cntr,1] = amp_src[j]
            # dur_amp_shift_id[cntr,2] = k
            rgb = mean_src*np.ones((timeBin_src))
            rgb[:dur_src[i]] = amp_src[j]
            shift = int((timeBin_src - dur_src[i])/2)
            rgb = np.roll(rgb,shift)
            step_block[:,cntr] = rgb
            
    n_reps = int(np.ceil(lum_obj_temp.shape[0]/total_conds)) #int(np.ceil(N_perturb/step_block.shape[-1]))
    lum_src = np.tile(step_block,[1,n_reps])
    lum_src = np.moveaxis(lum_src,0,-1)
    lum_src = lum_src[:lum_obj_temp.shape[0],:]
               
    lum_src = gaussian_filter1d(lum_src,sigma=sigma,axis=-1)
    idx_rand = np.random.permutation(np.arange(lum_src.shape[0]))
    lum_src = lum_src[idx_rand,:]
    
    lum_src = np.reshape(lum_src,lum_obj.shape[0],order='C')
   
    # lum_obj_perturb = lum_obj*lum_src
    
    return lum_obj,lum_src

def obj_source_multi_old(totalTime,timeBin_obj=10,mean_obj=10,amp_obj=2,timeBin_src = 50,mean_src=10,amp_src=5,dur_src=50,sigma=1,temporal_width=0,frac_perturb=1):
    
    nsamps_obj =  int(totalTime*1000*temporal_width/timeBin_obj)
    lum_obj = np.random.normal(mean_obj,amp_obj,nsamps_obj)
    lum_obj = np.repeat(lum_obj,timeBin_obj)
    lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    
    lum_obj_temp = np.reshape(lum_obj,(int(lum_obj.shape[0]/temporal_width),int(temporal_width)),order='C')
    lum_obj_rever = np.reshape(lum_obj_temp,lum_obj.shape[0],order='C')
    assert np.sum(abs(lum_obj-lum_obj_rever))==0
    
    lum_obj = lum_obj_temp

    
    range_shift = np.arange((2*timeBin_obj),timeBin_src-dur_src.max()-(2*timeBin_obj),timeBin_obj,dtype='int32')

    total_conds = dur_src.size * amp_src.size * range_shift.size
    step_block = np.zeros((timeBin_src,total_conds))
    dur_amp_shift_id = np.zeros((total_conds,3))
    cntr = -1
    for i in range(dur_src.size):
        for j in range(amp_src.size):
            for k in range_shift:
                cntr+=1
                dur_amp_shift_id[cntr,0] = dur_src[i]
                dur_amp_shift_id[cntr,1] = amp_src[j]
                rgb = mean_src*np.ones((timeBin_src))
                rgb[:dur_src[i]] = amp_src[j]
                shift = int((temporal_width - dur_src[i])/2)
                shift = int(temporal_width - dur_src[i])
                rgb = np.roll(rgb,shift)
                # rgb = np.roll(rgb,k)
                step_block[:,cntr] = rgb
                dur_amp_shift_id[cntr,2] = shift

            
    n_reps = int(np.ceil(lum_obj.shape[0]/total_conds)) #int(np.ceil(N_perturb/step_block.shape[-1]))
    lum_src = np.tile(step_block,[1,n_reps])
    lum_src = np.moveaxis(lum_src,0,-1)
    lum_src = lum_src[:lum_obj.shape[0],:]
               
    lum_src = gaussian_filter1d(lum_src,sigma=sigma,axis=-1)
    idx_rand = np.random.permutation(np.arange(lum_src.shape[0]))
    lum_src = lum_src[idx_rand,:]
   
    lum_obj_perturb = lum_obj*lum_src
    
    return lum_obj,lum_src,lum_obj_perturb

            
# %% old
    # lum_obj = np.random.normal(mean_obj,amp_obj,int(totalTime*60*1000/timeBin_obj))
    # lum_obj = np.repeat(lum_obj,timeBin_obj)
    # lum_obj = gaussian_filter(lum_obj,sigma=sigma)
    
    # mean_src = 5
    # timeBin_src = np.array([200,300,400,500,600,700,800]) #500
    # dur_src = np.array([10,20,30,40]) #50 
    # amp_src = np.array([10,20,30,40])
    # step_block = mean_src*np.ones((timeBin_src,amp_src.size))
    # step_block[:dur_src,:] = amp_src
    # # step_block = np.concatenate((step_block,amp_src+mean_src*np.ones(timeBin_src)),axis=0)
    # n_reps = int(np.ceil(lum_obj.shape[0]/step_block.shape[0]))
    # step_block_seq = step_block[:,0].copy()
    # for i in range(n_reps):
    #     which_amp = np.random.choice(np.arange(amp_src.size))
    #     step_block_seq = np.concatenate((step_block_seq,step_block[:,which_amp]),axis=0)
    # # step_block_seq = np.tile(step_block,n_reps)
    # lum_src = step_block_seq[:lum_obj.shape[0]]
    # lum_src = gaussian_filter(lum_src,sigma=sigma)



