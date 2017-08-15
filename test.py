#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:28:48 2017

@author: mahyar
"""
import numpy as np
import apollocaffe
from apollocaffe.layers import *
from pylayers import PyBackMultLayer, PyWeightedMeanLoss

### blobs
idata = 'actor_input_data'
output = 'actor_output_data'
softmax = 'actor_softmax'
vdata = 'value_func_data'
cdata = 'chosen_action_data'
vlayer = 'value_func_layer'
loss = 'loss'
bn = 'batchnorm'
gt = 'gt'

### params
wi = 'ip_weights'
bi = 'ip_bias'

input_data = np.array([[1.,2.,3.], [5.,4.,5.]])
value_data = np.array([10.,20.])
gt_data = np.array([0, 1])
norm = 1.0
net = apollocaffe.ApolloNet()
'''
net.f(NumpyData(idata, input_data))
net.f(NumpyData(cdata, gt_data))
net.f(NumpyData(vdata, value_data))
'''

'''
net.f(InnerProduct(output, 1, bottoms=[idata], param_names=[wi, bi], weight_filler=Filler('constant', 1.0)))
net.f(Softmax(softmax, bottoms=[output]))
net.f(PyBackMultLayer(vlayer, bottoms=[output, vdata], back_weight=norm))
net.f(SoftmaxWithLoss(loss, bottoms=[vlayer, cdata]))
'''
target = np.zeros(3) + 1
#input_data = -0.5*np.ones(3)
input_data = np.array([10.,20.,30.])
input_data = input_data.reshape((3,1))
net.f(NumpyData(idata, input_data))
net.f(NumpyData(gt, target))
net.f(InnerProduct(output, 1, bottoms=[idata], param_names=[wi, bi],
                   weight_filler=Filler('constant', 1.0), bias_filler=Filler('constant', 0.0)))
#net.f(SigmoidCrossEntropyLoss(loss, bottoms=[output, gt], loss_weight=1.0))
net.f(PyWeightedMeanLoss(loss, bottoms=[output, gt], loss_weight=1.0))
#net.f(BatchNorm(bn, bottoms=[idata], use_global_stats = False, moving_average_fraction=0.1, param_lr_mults=[0., 0., 0.],  param_names=['b0','b1','b2']))
#net.f()
#net.f(InnerProduct(output, 1, bottoms=[bn], param_names=[wi, bi], weight_filler=Filler('constant', 1.0)))
#net.f(SoftmaxWithLoss(loss, bottoms=[output, cdata]))

#net.backward()