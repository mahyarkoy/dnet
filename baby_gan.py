#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 16:18:14 2017

@author: mahyar
"""

import numpy as np
from collections import defaultdict, namedtuple
import apollocaffe
from apollocaffe.layers import *
import adam, adadelta
from pylayers import PyBackMultLayer, PyWeightedMeanLoss, PyHellingerLoss, PyLeastSquareLoss
import os

apollocaffe.set_device(0)
apollocaffe.set_random_seed(0)
np.random.seed(0)
OptConfig = namedtuple('OptConfig', 'rho eps lr clip')

class BabyGAN:
    def __init__(self, data_dim=2):
        ### network initializations
        self.net = apollocaffe.ApolloNet()
        self.par_inits = dict()

        ### optimizer parameters
        self.update_type = 'adam'
        if self.update_type == 'adam':
            self.d_state = adam.State()
            self.d_config = OptConfig(rho=0.5, eps=1e-8, lr=0.0001, clip=10.0)
            self.g_state = adam.State()
            self.g_config = OptConfig(rho=0.5, eps=1e-8, lr=0.0001, clip=10.0)
        else:
            self.d_state = adadelta.State()
            self.d_config = OptConfig(rho=0.95, eps=1e-6, lr=1.0, clip=10.0)
            self.g_state = adadelta.State()
            self.g_config = OptConfig(rho=0.95, eps=1e-6, lr=1.0, clip=10.0)
        
        ### network names
        self.d_var_name = 'd_var_name'
        self.d_name_real = 'd_name_real'
        self.d_name_gen = 'd_name_gen'
        self.d_loss_name_real = 'd_loss_name_real'
        self.d_loss_name_gen = 'd_loss_name_gen'

        self.g_var_name = 'g_var_name'
        self.g_name = 'g_name'
        self.g_loss_name = 'g_loss_name'

        ### network parameters
        self.z_dim = 256
        self.z_range = 1.0
        self.data_dim = data_dim
        self.d_loss_type = 'log'
        self.g_loss_type = 'mod'
        self.d_act = 'tanh'
        self.g_act = 'tanh'

        ### Fisher parametes
        self.fisher_info_list = list()
        self.param_history = list()
        self.max_memory = 5
        self.gen_confs = np.zeros(self.max_memory)
        self.gen_trace = np.zeros(self.max_memory)

    '''
    Forward the discriminator with the given name
    Input: x_data: name of NumpyData layer
    Retrun logit
    '''
    def d_forward(self, name, var_name, bottom, phase='train'):
        scale_layer = 'scale_layer_%s' % name
        hlayer_1 = 'd_hlayer_1_%s' % name
        relu_1 = 'd_relu_1_%s' % name
        hlayer_2 = 'd_hlayer_2_%s' % name
        relu_2 = 'd_relu_2_%s' % name
        hlayer_3 = 'd_hlayer_3_%s' % name
        relu_3 = 'd_relu_3_%s' % name
        hlayer_4 = 'd_hlayer_4_%s' % name
        relu_4 = 'd_relu_4_%s' % name
        logit = 'd_logit_%s' % name
        bn_1 = 'bn_1_%s_%s' % (name,phase)
        bn_2 = 'bn_2_%s_%s' % (name,phase)

        hw_1 = 'd_hw_1_%s' % var_name
        hb_1 = 'd_hb_1_%s' % var_name
        bn_m_1 = 'bn_m_1_%s' % var_name
        bn_v_1 = 'bn_v_1_%s' % var_name
        bn_c_1 = 'bn_c_1_%s' % var_name
        
        hw_2 = 'd_hw_2_%s' % var_name
        hb_2 = 'd_hb_2_%s' % var_name
        bn_m_2 = 'bn_m_2_%s' % var_name
        bn_v_2 = 'bn_v_2_%s' % var_name
        bn_c_2 = 'bn_c_2_%s' % var_name

        hw_3 = 'd_hw_3_%s' % var_name
        hb_3 = 'd_hb_3_%s' % var_name
        bn_m_3 = 'bn_m_3_%s' % var_name
        bn_v_3 = 'bn_v_3_%s' % var_name
        bn_c_3 = 'bn_c_3_%s' % var_name

        hw_4 = 'd_hw_4_%s' % var_name
        hb_4 = 'd_hb_4_%s' % var_name
        bn_m_4 = 'bn_m_4_%s' % var_name
        bn_v_4 = 'bn_v_4_%s' % var_name
        bn_c_4 = 'bn_c_4_%s' % var_name
        
        w = 'd_w_%s' % var_name
        b = 'd_b_%s' % var_name
        
        h_size = 128
        net = self.net
        global_stats = True if phase == 'test' else False
        avg_momentum = 0.0 if phase == 'eval' else 0.95

        #net.f(Power(scale_layer, bottoms=[bottom], scale=0.25))
        scale_layer = bottom
        net.f(InnerProduct(hlayer_1, h_size, bottoms=[scale_layer], param_names=[hw_1, hb_1],
            weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        if self.d_act == 'relu':
            net.f(ReLU(relu_1, bottoms=[hlayer_1], negative_slope=0.2))
        else:
            net.f(TanH(relu_1, bottoms=[hlayer_1]))

        net.f(InnerProduct(hlayer_2, h_size, bottoms=[relu_1], param_names=[hw_2, hb_2],
            weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_2, bottoms=[hlayer_2], param_names=['bn0', 'bn1', 'bn2'],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum))
        if self.d_act == 'relu':
            net.f(ReLU(relu_2, bottoms=[hlayer_2], negative_slope=0.2))
        else:
            net.f(TanH(relu_2, bottoms=[hlayer_2]))

        #net.f(InnerProduct(hlayer_3, h_size, bottoms=[relu_2], param_names=[hw_3, hb_3],
        #    weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        #net.f(ReLU(relu_2, bottoms=[hlayer_2]))
        #net.f(TanH(relu_3, bottoms=[hlayer_3]))

        #net.f(InnerProduct(hlayer_4, h_size, bottoms=[relu_3], param_names=[hw_4, hb_4],
        #    weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        #net.f(ReLU(relu_2, bottoms=[hlayer_2]))
        #net.f(TanH(relu_4, bottoms=[hlayer_4]))

        net.f(InnerProduct(logit, 1, bottoms=[relu_2], param_names=[w, b],
            weight_filler=Filler("xavier")))
        return logit
    
    '''
    Forward the generator with random input
    Input: name of NumpyData layer containing random numbers (bottom)
    Return: data
    '''
    def g_forward(self, name, var_name, bottom, phase='train'):
        hlayer_1 = 'g_hlayer_1_%s' % name
        relu_1 = 'g_relu_1_%s' % name
        hlayer_2 = 'g_hlayer_2_%s' % name
        relu_2 = 'g_relu_2_%s' % name
        hlayer_3 = 'g_hlayer_3_%s' % name
        relu_3 = 'g_relu_3_%s' % name
        hlayer_4 = 'g_hlayer_4_%s' % name
        relu_4 = 'g_relu_4_%s' % name
        data = 'g_data_%s' % name
        sig = 'g_sig_%s' % name
        bn_1 = 'bn_1_%s_%s' % (name,phase)
        bn_2 = 'bn_2_%s_%s' % (name,phase)
        
        hw_1 = 'g_hw_1_%s' % var_name
        hb_1 = 'g_hb_1_%s' % var_name
        bn_m_1 = 'bn_m_1_%s' % var_name
        bn_v_1 = 'bn_v_1_%s' % var_name
        bn_c_1 = 'bn_c_1_%s' % var_name

        hw_2 = 'g_hw_2_%s' % var_name
        hb_2 = 'g_hb_2_%s' % var_name
        bn_m_2 = 'bn_m_2_%s' % var_name
        bn_v_2 = 'bn_v_2_%s' % var_name
        bn_c_2 = 'bn_c_2_%s' % var_name

        hw_3 = 'g_hw_3_%s' % var_name
        hb_3 = 'g_hb_3_%s' % var_name
        bn_m_3 = 'bn_m_3_%s' % var_name
        bn_v_3 = 'bn_v_3_%s' % var_name
        bn_c_3 = 'bn_c_3_%s' % var_name

        hw_4 = 'g_hw_4_%s' % var_name
        hb_4 = 'g_hb_4_%s' % var_name
        bn_m_4 = 'bn_m_4_%s' % var_name
        bn_v_4 = 'bn_v_4_%s' % var_name
        bn_c_4 = 'bn_c_4_%s' % var_name

        w = 'g_w_%s' % var_name
        b = 'g_b_%s' % var_name
        
        h_size = 128
        net = self.net

        global_stats = True if phase == 'test' else False
        avg_momentum = 0.0 if phase == 'eval' else 0.95
        net.f(InnerProduct(hlayer_1, h_size, bottoms=[bottom], param_names=[hw_1, hb_1],
            weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        if self.g_act == 'relu':
            net.f(ReLU(relu_1, bottoms=[hlayer_1]))
        else:
            net.f(TanH(relu_1, bottoms=[hlayer_1]))
        
        net.f(InnerProduct(hlayer_2, h_size, bottoms=[relu_1], param_names=[hw_2, hb_2],
            weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        if self.g_act == 'relu':
            net.f(ReLU(relu_2, bottoms=[hlayer_2]))
        else:
            net.f(TanH(relu_2, bottoms=[hlayer_2]))

        #net.f(InnerProduct(hlayer_3, h_size, bottoms=[relu_2], param_names=[hw_3, hb_3],
        #    weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        #net.f(ReLU(relu_2, bottoms=[hlayer_2]))
        #net.f(TanH(relu_3, bottoms=[hlayer_3]))

        #net.f(InnerProduct(hlayer_4, h_size, bottoms=[relu_3], param_names=[hw_4, hb_4],
        #    weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        #net.f(BatchNorm(bn_1, bottoms=[hlayer_1], param_names=[bn_m_1, bn_v_1, bn_c_1],
        #    use_global_stats = global_stats, moving_average_fraction=avg_momentum, param_lr_mults=[0.,0.,0.]))
        #net.f(ReLU(relu_2, bottoms=[hlayer_2]))
        #net.f(TanH(relu_4, bottoms=[hlayer_4]))
        
        net.f(InnerProduct(data, self.data_dim , bottoms=[relu_2], param_names=[w, b],
            weight_filler=Filler("xavier"), bias_filler=Filler("constant", 0.0)))
        
        return data
    
    '''
    Calculate gt*logp + (1-gt)*(1-logp)
    Input: name of logits layer (bottom), ground truth labels (gt), loss_weight (weight)
    Output: loss value
    '''
    def log_loss(self, name, bottom, target, loss_weight, loss_type='log'):
        gt = 'loss_gt_%s' % name
        loss = 'loss_val_%s' % name
        sig = 'sig_val_%s' % name
        
        net = self.net
        if target == 0:
            target = -1 if loss_type == 'was' else 0
        target_array = target * np.ones(net.blobs[bottom].shape)
        net.f(NumpyData(gt, target_array))
        if loss_type == 'was':
            net.f(PyWeightedMeanLoss(loss, bottoms=[bottom, gt], loss_weight=loss_weight))
        elif loss_type == 'hel':
            net.f(Sigmoid(sig, bottoms=[bottom]))
            net.f(PyHellingerLoss(loss, bottoms=[sig], loss_weight=loss_weight))
        elif loss_type == 'lsq':
            net.f(Sigmoid(sig, bottoms=[bottom]))
            net.f(PyLeastSquareLoss(loss, bottoms=[sig, gt], loss_weight=loss_weight))
        else:
            net.f(SigmoidCrossEntropyLoss(loss, bottoms=[bottom, gt], loss_weight=loss_weight))
        return loss

    '''
    Evaluate discrimnator for either real or gen inputs (real has priority)
    Input: name of real and z layer, batch_size
    Output: logs of acc, loss, average logit, MS logit_diff, MS param_diff given real or gen inputs
    '''
    def d_eval_step(self, r_layer, z_layer, batch_size):
        net = self.net
        if r_layer:
            ### forward discriminator and get real logits: batch_size*1
            u_logit = self.d_forward(self.d_name_real, self.d_var_name, r_layer, phase='eval')
            u_loss = self.log_loss(self.d_loss_name_real, u_logit, 1.0, loss_weight=1.0, loss_type=self.d_loss_type)
            acc_sign = 1.0
        elif z_layer:
            ### forward generator and get the generated layer in data: batch_size*data_dim
            u_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='eval')
            ### forward discriminator and get real logits: batch_size*1
            u_logit = self.d_forward(self.d_name_gen, self.d_var_name, u_layer, phase='eval')
            u_loss = self.log_loss(self.d_loss_name_gen, u_logit, 0.0, loss_weight=1.0, loss_type=self.d_loss_type)
            acc_sign = -1.0
        else:
            raise ValueError('In d_eval_step: either z_layer or r_layer should have value other than None!')

        net.backward()
        ### logs
        u_acc = np.mean(acc_sign * net.blobs[u_logit].data > 0)
        u_loss = net.blobs[u_loss].data.item()
        u_logit_data = np.mean(net.blobs[u_logit].data)
        u_logit_diff = np.mean(np.square(net.blobs[u_logit].diff)) ** 0.5
        param_sum = 0.0
        count = 0.0
        for param_name in net.active_param_names():
            if param_name.startswith('d_'):
                param = net.params[param_name]
                grad = param.diff * net.param_lr_mults(param_name)
                param_sum += np.sqrt(np.mean(np.square(grad)))
                count += 1
        u_param_diff = 1.0 * param_sum / count
        return [u_acc, u_loss, u_logit_data, u_logit_diff, u_param_diff]
        
    '''
    Evaluate discriminator on both real and gen inputs, then update the params of discriminator
    Inputs: name of the real and z layers, batch_size
    Output: two list each containing the logs of the discriminator given real and gen inputs
    '''
    def d_one_step(self, r_layer, z_layer, batch_size, update=False, phase='train'):
        net = self.net
        self.clean_network()
        ### forward eval on real data
        d_r_acc, d_r_loss, d_r_logit_data, d_r_logit_diff, d_r_param_diff =\
            self.d_eval_step(r_layer, None, batch_size)
        
        self.clean_network()
        ### forward eval on gen data
        d_g_acc, d_g_loss, d_g_logit_data, d_g_logit_diff, d_g_param_diff =\
            self.d_eval_step(None, z_layer, batch_size)
        
        if phase == 'train':
            self.clean_network()

            ### forward generator and get the generated layer in data: batch_size*data_dim
            g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='train')
            ### forward discriminator and get real/gen logits: batch_size*1
            r_logit = self.d_forward(self.d_name_real, self.d_var_name, r_layer, phase='train')
            g_logit = self.d_forward(self.d_name_gen, self.d_var_name, g_layer, phase='train')
            
            ### get losses and update discriminator variables only
            d_r_loss = self.log_loss(self.d_loss_name_real, r_logit, 1.0, loss_weight=1.0, loss_type=self.d_loss_type)
            d_g_loss = self.log_loss(self.d_loss_name_gen, g_logit, 0.0, loss_weight=1.0, loss_type=self.d_loss_type)
            
            d_r_loss = net.blobs[d_r_loss].data.item()
            d_g_loss = net.blobs[d_g_loss].data.item()

            ### save initialized values of all parameters in the network
            if len(self.par_inits) == 0:
                self.save_init_network()

            net.backward()

        ### update parameters with adam (no clipping)
        if update:
            if self.update_type == 'adam':
                adam.update(net, self.d_state, self.d_config, 'd_', self.d_loss_type)
            else:
                adadelta.update(net, self.d_state, self.d_config, 'd_', self.d_loss_type)

        #return ([d_r_acc, d_r_loss, d_r_logit_data, d_r_logit_diff, d_r_param_diff],
        #        [d_g_acc, d_g_loss, d_g_logit_data, d_g_logit_diff, d_g_param_diff])
        return ([d_r_loss, d_r_logit_data, d_r_logit_diff, d_r_param_diff],
                [d_g_loss, d_g_logit_data, d_g_logit_diff, d_g_param_diff])
         
    '''
    Evaluate generator on gen inputs, then update the params of generator
    Inputs: name of the z layers, batch_size
    Output: a list containing the logs of the generator given gen inputs
    '''
    def g_one_step(self, z_layer, batch_size, update=False, phase='train'):
        net = self.net
        self.clean_network()
        sig = 'd_sig_logit'
        
        ### forward generator and get the generated layer in data: batch_size*data_dim
        g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase=phase)
        ### forward discriminator and get gen logits: batch_size*1
        g_logit = self.d_forward(self.d_name_gen, self.d_var_name, g_layer, phase=phase)
        ### get loss and update generator variables only (argmax log D(G(z)) )
        ### >>> HELLINGER CHANGE HERE
        if self.g_loss_type == 'log':
            g_loss = self.log_loss(self.g_loss_name, g_logit, 0.0, loss_weight=-1.0, loss_type=self.g_loss_type)
        else: ## mod was hel
            g_loss = self.log_loss(self.g_loss_name, g_logit, 1.0, loss_weight=1.0, loss_type=self.g_loss_type)
        net.backward()
        
        ### logs
        d_g_acc = np.mean(net.blobs[g_logit].data < 0)
        g_loss = net.blobs[g_loss].data.item()
        g_logit_data = np.mean(net.blobs[g_logit].data)
        g_logit_diff = np.mean(np.square(net.blobs[g_logit].diff)) ** 0.5
        g_out_diff = np.mean(np.square(net.blobs[g_layer].diff)) ** 0.5
        param_sum = 0.0
        count = 0.0
        for param_name in net.active_param_names():
            if param_name.startswith('g_'):
                param = net.params[param_name]
                grad = param.diff * net.param_lr_mults(param_name)
                param_sum += np.sqrt(np.mean(np.square(grad)))
                count += 1
        g_param_diff = 1.0 * param_sum / count
        
        ### update parameters with adam (no clipping)
        if update:
            if self.update_type == 'adam':
                adam.update(net, self.g_state, self.g_config, 'g_', self.g_loss_type)
            else:
                adadelta.update(net, self.g_state, self.g_config, 'g_', self.g_loss_type)
            #adadelta.update(net, self.g_state, self.g_config, 'g_', self.g_loss_type,
            #    self.gen_confs, self.gen_trace, self.fisher_info_list, self.param_history)
            self.clean_network()
            g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='eval')
        
        #return ([d_g_acc, g_loss, g_logit_data, g_logit_diff, g_out_diff, g_param_diff], net.blobs[g_layer].data)
        return ([g_loss, g_logit_diff, g_out_diff, g_param_diff], net.blobs[g_layer].data)
    
    '''
    If batch_data is not None: train disc and eval fixed gen and disc for one step
    If gen_update is True: train and eval gen for one step
    If dis_only is True: only get the logits for the disc given the batch_data 
    '''   
    def step(self, batch_data, batch_size, gen_update=False, dis_only=False, gen_only=False):
        self.clean_network()
        net = self.net
        r_layer = 'r_layer'
        z_layer = 'z_layer'
        
        ### make real data layer
        if batch_data is not None:
            net.f(NumpyData(r_layer, batch_data))
            ### do discrimnation on batch_data and return logits ndarray
            if dis_only:
                u_logit = self.d_forward(self.d_name_real, self.d_var_name, r_layer, phase='test')
                return net.blobs[u_logit].data.flatten()
        
        ### sample z from uniform (-1,1)
        z_data = np.random.uniform(low=-self.z_range, high=self.z_range, size=(batch_size, self.z_dim))
        #z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))
        net.f(NumpyData(z_layer, z_data))
        if gen_only:
            g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='eval')
            return net.blobs[g_layer].data

        ### run one training step on discriminator if batch_data is not None, otherwise on generator
        if not gen_update:
            g_logs, g_data = self.g_one_step(z_layer, batch_size, update=False, phase='eval')
            d_r_logs, d_g_logs = self.d_one_step(r_layer, z_layer, batch_size, update=True)
            logs = (g_logs, d_r_logs, d_g_logs)
        else:
            if batch_data is not None:
                d_r_logs, d_g_logs = self.d_one_step(r_layer, z_layer, batch_size, update=False, phase='eval')
            g_logs, g_data = self.g_one_step(z_layer, batch_size, update=True, phase='train')
            logs = (g_logs, d_r_logs, d_g_logs) if batch_data is not None else (g_logs, None, None)

        return logs, g_data

    def save(self, fname):
        self.net.save(fname)

    def load(self, fname):
        self.net.load(fname)
    '''
    Updates stored fisher information variables which are used in gen_updates to consolidate previous gens
    Returns the current gen_conf
    Use it like a check point when you want to consolidate a desirable gen weights
    '''
    def gen_consolidate(self, count=50):
        net = self.net
        self.clean_network()
        z_layer = 'z_layer'
        sig_layer = 'sig_layer'
        unit_loss = 'unit_loss'
        gen_conf = 0.0
        fisher_info = dict()

        ### check confidence of generator: replace the previous gen with lowest confidence
        z_data = np.random.uniform(low=-1.0, high=1.0, size=(count, self.z_dim))
        net.f(NumpyData(z_layer, z_data))
        g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='eval')
        g_logit = self.d_forward(self.d_name_gen, self.d_var_name, g_layer, phase='eval')
        net.f(Sigmoid(sig_layer, bottoms=[g_logit]))
        
        ### mean_z sig(D(G(z))) as the confidence/usefulness of current generator
        gen_conf = max(np.mean(net.blobs[sig_layer].data) * 2, 0.99)
        dis_confs = self.gen_confs ** self.gen_trace
        min_conf_id = np.argmin(dis_confs)
        if gen_conf < self.dis_confs[min_conf_id]: ## NOTE: correct because gen_confs inits to 0
            return

        ### mean_z gen_param_diff^2 as a measure of importance of each param in current gen
        for n in range(count):
            self.clean_network()
            z_data = np.random.uniform(low=-1.0, high=1.0, size=(1, self.z_dim))
            net.f(NumpyData(z_layer, z_data))
            g_layer = self.g_forward(self.g_name, self.g_var_name, z_layer, phase='eval')
            g_logit = self.d_forward(self.d_name_gen, self.d_var_name, g_layer, phase='eval')
            net.f(PyUnitLoss(unit_loss, bottoms=[g_logit]))
            net.backward()
            ### collect fisher info
            for p in net.active_param_names():
                if not p.startswith('g_'):
                    continue
                if n == 0:
                    fisher_info[p] = net.params[p].diff ** 2
                    param_data[p] = net.params[p].data
                else:
                    fisher_info[p] += net.params[p].diff ** 2
        for p in fisher_info.keys():
            fisher_info[p] /= 1.0 * count

        memory_len = len(self.fisher_info_list)
        if memory_len == self.max_memory:
            self.gen_confs[min_conf_id] = gen_conf
            self.gen_trace[min_conf_id] = 0.0
            self.fisher_info_list[min_conf_id] = fisher_info
            self.param_history[min_conf_id] = param_data
        else:
            self.gen_confs[memory_len] = gen_conf
            self.gen_trace[memory_len] = 0.0 ## Redundant, for clarity
            self.fisher_info_list.append(fisher_info)
            self.param_history.append(param_data)
        self.gen_trace += 1
        return gen_conf, dis_confs, self.gen_trace

    def clean_network(self):
        net = self.net
        for par in net.active_param_names():
            net.params[par].diff[...] = 0.0
        net.clear_forward()

    def reset_network(self, key):
        net = self.net
        for par in net.params.keys():
            if par.startswith(key):
                net.params[par].data[...] = self.par_inits[par]
                if key == 'd_':
                    del self.d_state.sq_grads[par]
                else:
                    del self.g_state.sq_grads[par]

    def save_init_network(self):
        net = self.net
        for par in net.params.keys():
            self.par_inits[par] = np.array(net.params[par].data)




        
        
        
        
        
        
        
        
        
        
