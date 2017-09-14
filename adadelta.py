#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 17:07:56 2017

@author: mahyar
based on nmn code by jacob andreas
"""

from collections import namedtuple
import numpy as np

class State:
    def __init__(self):
        self.sq_updates = dict()
        self.sq_grads = dict()
    
    def save(self, filename):
        np.savez(filename+'_sq_updates.npz', **self.sq_updates)
        np.savez(filename+'_sq_grads.npz', **self.sq_grads)

    def load(self, filename):
        updates_ld = np.load(filename+'_sq_updates.npz')
        grads_ld = np.load(filename+'_sq_grads.npz')
        self.sq_updates = dict(updates_ld.items())
        self.sq_grads = dict(grads_ld.items())

def update(net, state, config, update_param_key=None, loss_type='log', confs=None, trace=None, fisher_list=None, param_history=None):
    rho = config.rho
    epsilon = config.eps
    lr = config.lr
    clip = config.clip
    if update_param_key is None:
        update_param_names = net.active_param_names()
    else:
        update_param_names = [n for n in net.active_param_names() if n.startswith(update_param_key)]

    all_norm = 0.
    for param_name in update_param_names:
        param = net.params[param_name]
        grad = param.diff * net.param_lr_mults(param_name)
        all_norm += np.sum(np.square(grad))
    all_norm = np.sqrt(all_norm)

    for param_name in update_param_names:
        param = net.params[param_name]
        ### elastic weight update
        threshold = 1e-5
        if update_param_key == 'g_' and confs is not None:
            dis_confs = confs ** trace
            for i, v in enumerate(dis_confs):
                if v < threshold:
                    continue
                elastic_grad = 2*v*fisher_list[i][param_name]*(param.data - param_history[i][param_name])
            grad = (param.diff+elastic_grad) * net.param_lr_mults(param_name)
        else:
            grad = param.diff * net.param_lr_mults(param_name)

        if all_norm > clip:
            grad = clip * grad / all_norm

        if param_name in state.sq_grads:
            state.sq_grads[param_name] = \
                (1 - rho) * np.square(grad) + rho * state.sq_grads[param_name]
            rms_update = np.sqrt(state.sq_updates[param_name] + epsilon)
            rms_grad = np.sqrt(state.sq_grads[param_name] + epsilon)
            update = rms_update / rms_grad * grad

            state.sq_updates[param_name] = \
                (1 - rho) * np.square(update) + rho * state.sq_updates[param_name]
        else:
            state.sq_grads[param_name] = (1 - rho) * np.square(grad)
            update = np.sqrt(epsilon) / np.sqrt(epsilon +
                    state.sq_grads[param_name]) * grad
            state.sq_updates[param_name] = (1 - rho) * np.square(update)


        param.data[...] -= lr * update
        param.diff[...] = 0.0
        if loss_type == 'was' and update_param_key == 'd_':
            weight_clip = 0.05
            param.data[param.data[...] > weight_clip] = weight_clip
            param.data[param.data[...] < -weight_clip] = -weight_clip

