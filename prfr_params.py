#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:29:42 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

# %% primate cones
def fr_cones_fixed():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_cones_trainable():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


# %% Primate rods

def fr_rods_fixed():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.25
    pr_params['beta_scaleFac'] = 100.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_rods_trainable():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.25
    pr_params['beta_scaleFac'] = 100.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params

