#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:17:52 2017

@author: mahyar
based on nmn code by jacob andreas
"""

from collections import namedtuple
import numpy as np

class State:
    def __init__(self):
        self.mean_grads = dict()
        self.sq_grads = dict()
    
    def save(self, filename):
        np.savez(filename+'_sq_updates.npz', **self.mean_grads)
        np.savez(filename+'_sq_grads.npz', **self.sq_grads)

    def load(self, filename):
        updates_ld = np.load(filename+'_mrean_grads.npz')
        grads_ld = np.load(filename+'_sq_grads.npz')
        self.mean_grads = dict(updates_ld.items())
        self.sq_grads = dict(grads_ld.items())

def update(net, state, config, update_param_key):
    rho = config.rho
    epsilon = config.eps
    lr = config.lr
    clip = config.clip
    update_param_names = [n for n in net.active_param_names() if n.startswith(update_param_key)]
    key_norm = 0.0
    off_key_norm = 0.0
    key_count = 0.0
    off_key_count = 0.0
    for param_name in net.active_param_names():
        if param_name.startswith(update_param_key):
            param = net.params[param_name]
            grad = param.diff * net.param_lr_mults(param_name)
            key_norm += np.sum(np.square(grad))
            key_count += 1
        else:
            param = net.params[param_name]
            grad = param.diff * net.param_lr_mults(param_name)
            off_key_norm += np.sum(np.square(grad))
            off_key_count += 1
    key_norm = key_norm / key_count
    off_key_norm = off_key_norm / off_key_count

    for param_name in update_param_names:
        param = net.params[param_name]
        grad = param.diff * net.param_lr_mults(param_name)

        #if all_norm > clip:
        #   grad = clip * grad / all_norm

        if param_name in state.sq_grads:
            state.sq_grads[param_name] = \
                rho * np.square(grad) + (1 - rho) * state.sq_grads[param_name]
            state.mean_grads[param_name] = \
                rho * grad + (1 - rho) * state.mean_grads[param_name]
            rms_grad = np.sqrt(state.sq_grads[param_name] + epsilon)
            update = -lr / rms_grad * state.mean_grads[param_name]            
        else:
            state.sq_grads[param_name] = rho * np.square(grad)
            state.mean_grads[param_name] = rho * np.square(grad)
            rms_grad = np.sqrt(state.sq_grads[param_name] + epsilon)
            update = -lr / rms_grad * state.mean_grads[param_name]

        param.data[...] += update
        param.diff[...] = 0.0
        
    return key_norm, off_key_norm