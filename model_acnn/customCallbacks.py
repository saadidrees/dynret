#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:18:15 2022

@author: saad
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
from collections import Iterable


class displayLog(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("LR - {}".format(self.model.optimizer.learning_rate))
        

class TerminateOnNaN(keras.callbacks.Callback):
    def __init__(self):
        super(TerminateOnNaN,self).__init__()
        
    def on_batch_end(self,batch,logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True


class NBatchCSVLogger(keras.callbacks.Callback):
    """Callback that streams every batch results to a csv file.
    """
    def __init__(self, filename, path_model_save, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.path_model_save = path_model_save
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if os.name == 'nt' else ''
        super(NBatchCSVLogger, self).__init__()
        
    def on_epoch_begin(self, epoch, logs=None):
        filename = self.filename +'_epoch-%03d.csv' %epoch
        if self.append:
            if os.path.exists(filename):
                with open(filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(filename, 'w' + self.file_flags)
        
        self.epoch = epoch
            
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['LR'] = np.array(self.model.optimizer.learning_rate)
        
        # Weights
        weights_dict = {}
        for layer in self.model.layers:
            for weight in layer.weights:
                weights_mean = np.mean(weight)
                if np.isnan(weights_mean):
                    rgb = np.sum(np.isnan(weight))
                    weights_mean = 'nan-%d' %rgb
                weights_dict[weight.name] = weights_mean
        weights_keys = list(weights_dict.keys())
        
        dir_epoch = (os.path.join(self.path_model_save,'epoch_%03d' %self.epoch))
        if not os.path.exists(dir_epoch):
            os.mkdir(dir_epoch)
        fname_weights = os.path.join(dir_epoch,'weights-%03d.h5' % batch)
        self.model.save_weights(fname_weights)

        

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
            
        if self.keys is None:
            self.keys = list(logs.keys())
            
        if self.model.stop_training:
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
            
        if not self.writer:
            
            class CustomDialect(csv.excel):
                delimiter = self.sep
            self.writer = csv.DictWriter(self.csv_file,fieldnames=['batch'] + self.keys + weights_keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        
        row_dict = dict(batch=batch)
        row_dict.update((key, logs[key]) for key in self.keys)               
        row_dict.update((key, weights_dict[key]) for key in weights_keys)
        
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        
    def on_epoch_end(self, epoch,logs=None):
        self.csv_file.close()
        self.writer = None
        