#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:30:12 2017

@author: mahyar
"""

import numpy as np
from collections import defaultdict, namedtuple
import apollocaffe
from apollocaffe.layers import *
import adadelta
from pylayers import PyBackMultLayer
import os

OptConfig = namedtuple('OptConfig', 'rho eps lr clip')
class Dnet:
    def __init__(self):
        ### network containing subnets for RL decision making
        self.actor_net = apollocaffe.ApolloNet()
        ### network containing subnets for classificatoin actions
        self.action_net = apollocaffe.ApolloNet()
        ### archive of all action nets stored with ids
        self.net_archive = dict()
        self.net_archive_index = dict()
        self.chosen_actions = list()
        self.actor_state = adadelta.State()
        self.actor_config = OptConfig(rho=0.95, eps=1e-6, lr=1.0, clip=10.0)
        self.action_state = adadelta.State()
        self.action_config = OptConfig(rho=0.95, eps=1e-6, lr=1.0, clip=10.0)
        self.value_weights = None
        self.action_size = 4
        self.label_size = 2
        self.mu_list = list()
        self.std_list = list()
        self.count_list = list()
        self.value_counts = 0.0
        
    '''
    return net index stored in net_archive, add if does not exit
    '''
    def net_index(self, net_struct):
        if net_struct in self.net_archive:
            return self.net_archive[net_struct]
        else:
            net_count = len(self.net_archive_index)
            self.net_archive[net_struct] = net_count
            self.net_archive_index[net_count] = net_struct
            self.mu_list.append(None)
            self.std_list.append(None)
            self.count_list.append(None)
            return net_count
    
    '''
    run forward pass on actor network, choose net_structs as actions
    '''    
    def actor_forward(self, batch_data, greedy=False):
        ### blobs
        idata = 'actor_input_data'
        hlayer = 'actor_hidden'
        output = 'actor_output_data'
        softmax = 'actor_softmax'
        relu = 'actor_relu'
        
        ### params
        actorhw = 'actor_hidden_weights'
        actorhb = 'actor_hidden_bias'
        actorw = 'actor_weights'
        actorb = 'actor_bias'
        
        actor_hidden_size = 8
        ### action net structure
        net = self.actor_net
        net.f(NumpyData(idata, batch_data))
        net.f(InnerProduct(hlayer, actor_hidden_size, bottoms=[idata], param_names=[actorhw, actorhb]))
        net.f(ReLU(relu, bottoms=[hlayer]))
        net.f(InnerProduct(output, self.action_size, bottoms=[relu], param_names=[actorw, actorb]))
        #net.f(InnerProduct(output, self.action_size, bottoms=[idata], param_names=[actorw, actorb]))
        net.f(Softmax(softmax, bottoms=[output]))
        probs = net.blobs[softmax].data
        if greedy:
            self.chosen_actions = np.argmax(probs,axis=1)
        else:
            self.chosen_actions = \
                np.array([np.random.choice(range(self.action_size), p=probs[i, :]) for i in range(len(batch_data))])
        
        ### manual chosen_action, comment to use the actions chosen by RL agent
        ca = list()
        for x in batch_data[:,0]:
            if x < -1:
                ca.append(0)
            #else:
            #    ca.append(1)
            elif x < 0:
                ca.append(1)
            elif x < 1:
                ca.append(2)
            else:
                ca.append(3)
        self.chosen_actions = np.array(ca)
        
        ### convert chosen_action to network structures
        net_structs = [str(a) for a in self.chosen_actions]
        return net_structs
    
    '''
    run forward pass on action networks for classification, construct loss
    '''
    def action_net_forward(self, net_id, net_data, net_gt, phase):
        # construct nets and collect returns
        ### blobs
        idata = 'input_data_%d' % net_id
        gt = 'gt_data_%d' % net_id
        hlayer = 'hidden_layer_%d' % net_id
        h2layer = 'hidden2_layer_%d' % net_id
        output = 'output_data_%d' % net_id
        softmax = 'softmax_%d' % net_id
        loss = 'loss_%d' % net_id
        invmask = 'invmask_%d' % net_id
        invout = 'invout_%d' % net_id
        tile = 'tile_%d' % net_id
        relu = 'relu_%d' % net_id
        relu2 = 'relu2_%d' % net_id
        dp = 'dp_%d' % net_id
        dp2 = 'dp2_%d' % net_id
        
        ### params
        hw = 'hidden_weights_%d' % net_id
        hb = 'hidden_bias_%d' % net_id
        h2w = 'hidden2_weights_%d' % net_id
        h2b = 'hidden2_bias_%d' % net_id
        ow = 'output_weights_%d' % net_id
        ob = 'output_bias_%d' % net_id
        
        ### init mean and std normalizations
        if self.mu_list[net_id] is None:
            self.mu_list[net_id] = np.zeros(net_data.shape[1])
            self.std_list[net_id] = np.ones(net_data.shape[1])
            self.count_list[net_id] = 0.0

        self.count_list[net_id] += 1.0
        self.mu_list[net_id] += 0.1 * (np.mean(net_data,axis=0) - self.mu_list[net_id]) / self.count_list[net_id]
        self.std_list[net_id] += 0.1 * (np.std(net_data,axis=0) - self.std_list[net_id]) / self.count_list[net_id]
        if any(self.std_list[net_id] == 0):
            print 'mu>>> ', self.mu_list[net_id]
            print 'std>>> ', self.std_list[net_id]
            print 'net_id>>> ', self.net_archive_index[net_id]
            print net_data
            
        net_data = net_data / self.std_list[net_id] - self.mu_list[net_id]
        
        hidden_size = 128
        ### action net structure
        net = self.action_net
        net_struct = self.net_archive_index[net_id]
        net.f(NumpyData(idata, net_data))
        
        net.f(InnerProduct(hlayer, hidden_size, bottoms=[idata], param_names=[hw, hb]))
        net.f(ReLU(relu, bottoms=[hlayer]))
        #if phase == 'train':
        #    net.f(Dropout(dp, 0.5, bottoms=[relu]))
        #    hlayer_final = dp
        #else:
        #    hlayer_final = relu
        hlayer_final = relu
        net.f(InnerProduct(h2layer, hidden_size, bottoms=[hlayer_final], param_names=[h2w, h2b]))
        net.f(ReLU(relu2, bottoms=[h2layer]))
        #if phase == 'train':
        #    net.f(Dropout(dp2, 0.5, bottoms=[relu2]))
        #    h2layer_final = dp2
        #else:
        #    h2layer_final = relu2
        h2layer_final = relu2
        net.f(InnerProduct(output, self.label_size, bottoms=[h2layer_final], param_names=[ow, ob]))
        
        #net.f(InnerProduct(output, self.label_size, bottoms=[idata], param_names=[ow, ob]))
        
        #net.f(Tile(tile, axis=1, tiles=2, bottoms=[output]))
        #invmask_data = np.ones(net.blobs[tile].shape)
        #invmask_data[:,1] *= -1
        #net.f(NumpyData(invmask, invmask_data))
        #net.f(Eltwise(invout, 'PROD', bottoms=[tile, invmask]))
        net.f(Softmax(softmax, bottoms=[output]))
        probs = net.blobs[softmax].data
        preds = np.argmax(probs, axis=1)

        ### return predictions, probabilities of preds, and average rewards (-loss)
        ### rewards is None in no ground truth
        row_index = range(probs.shape[0])
        if net_gt is None:
            rewards = None
        else:
            rewards = np.log(probs[row_index,net_gt])
        if phase == 'train':
            net.f(NumpyData(gt, net_gt))
            net.f(SoftmaxWithLoss(loss, bottoms=[output, gt]))
        return preds, probs[row_index,preds], rewards
    
    '''
    load net structures into archive and get indicies.
    create a net dict to store list of corresponding data for that structure.
    create a dispatch dict for retaining data order.
    '''
    def dispatch(self, batch_data, net_structs):
        net_dict = defaultdict(list)
        dispatch_dict = defaultdict(list)
        net_ids = [self.net_index(s) for s in net_structs]
        for k, d in enumerate(batch_data):
            c = net_ids[k]
            net_dict[c].append(d)
            dispatch_dict[c].append(k)
        return net_dict, dispatch_dict
    
    '''
    update value function and actor network (policy gradient)
    '''
    def actor_update(self, rewards):
        ### blobs
        idata = 'actor_input_data'
        output = 'actor_output_data'
        softmax = 'actor_softmax'
        vdata = 'value_func_data'
        cdata = 'chosen_action_data'
        vlayer = 'value_func_layer'
        loss = 'actor_loss'
        relu = 'actor_relu'
        
        e0data = 'actor_e0_data'
        e1data = 'actor_e1_data'
        e2data = 'actor_e2_data'
        e3data = 'actor_e3_data'
        losse0 = 'actor_losse0'
        losse1 = 'actor_losse1'
        losse2 = 'actor_losse2'
        losse3 = 'actor_losse3'
        
        net = self.actor_net
        ### value function update
        vlearn_rate = 0.1
        features = net.blobs[relu].data
        features.reshape(features.shape[0], features.shape[1])
        if self.value_weights is None:
            self.value_weights = np.zeros(features.shape[1])
            #self.value_weights = 0.0
            self.value_counts = 0.0
        self.value_counts += 1
        value_delta = rewards - features.dot(self.value_weights)
        self.value_weights += vlearn_rate*value_delta.dot(features) / self.value_counts
        #value_delta = rewards - self.value_weights
        #self.value_weights += vlearn_rate*np.mean(value_delta) / self.value_counts
        
        ### policy update
        #norm = 1.0 / rewards.size
        norm = 1.0
        net.f(NumpyData(vdata, value_delta))
        net.f(NumpyData(cdata, self.chosen_actions))
        net.f(PyBackMultLayer(vlayer, bottoms=[output, vdata], back_weight=norm))
        net.f(SoftmaxWithLoss(loss, bottoms=[vlayer, cdata]))
        ### entropy
        '''
        net.f(NumpyData(e0data, np.zeros(self.chosen_actions.shape[0])))
        net.f(NumpyData(e1data, np.ones(self.chosen_actions.shape[0])))
        net.f(NumpyData(e2data, 2*np.ones(self.chosen_actions.shape[0])))
        net.f(NumpyData(e3data, 3*np.ones(self.chosen_actions.shape[0])))
        net.f(SoftmaxWithLoss(losse0, bottoms=[output, e0data], loss_weight=0.01))
        net.f(SoftmaxWithLoss(losse1, bottoms=[output, e1data], loss_weight=0.01))
        net.f(SoftmaxWithLoss(losse2, bottoms=[output, e2data], loss_weight=0.01))
        net.f(SoftmaxWithLoss(losse3, bottoms=[output, e3data], loss_weight=0.01))
        '''
        net.backward()
        adadelta.update(net, self.actor_state, self.actor_config)
        # should add entropy to loss
        return
    
    '''
    update action net for classification (backprop)
    '''
    def action_net_update(self):
        net = self.action_net
        net.backward()
        adadelta.update(net, self.action_state, self.action_config)
        return
    
    '''
    generate net structures, forward data, and update actor and action networks.
    return number of correct predictions, average reward
    '''
    def run_batch(self, batch_data, batch_gt, phase):
        ### choose structure and dispatch data to corresponding nets
        net_structs = self.actor_forward(batch_data, True if phase=='test' else False)
        net_dict, dispatch_dict = self.dispatch(batch_data, net_structs)
        ### execute nets and collect results: preds, probs, rewards
        batch_preds = np.zeros(batch_data.shape[0])
        batch_probs = np.zeros(batch_data.shape[0])
        batch_rewards = np.zeros(batch_data.shape[0])
        for net_id, net_data in net_dict.items():
            dispatch_ids = dispatch_dict[net_id]
            if batch_gt is None:
                net_gt = None
            else:
                net_gt = batch_gt[dispatch_ids].astype(int)
            preds, probs, rewards = self.action_net_forward(net_id, np.asarray(net_data), net_gt, phase)
            batch_preds[dispatch_ids] = preds
            batch_probs[dispatch_ids] = probs
            batch_rewards[dispatch_ids] = rewards
        ### update actor net and action nets
        if phase == 'train':
            self.actor_update(batch_rewards)
            self.action_net_update()
            
        ### return correct count, average rewards, predictions and probabilities for this batch
        if batch_gt is None:
            return None, None, batch_preds, batch_probs
        else:
            return np.sum(batch_gt - batch_preds == 0), np.sum(batch_rewards), batch_preds, batch_probs
    
    '''
    reset the net to apply the next forward pass
    '''    
    def reset(self):
        self.actor_net.clear_forward()
        self.action_net.clear_forward()
    
    '''
    run one epoch over the input data.
    return list of correct count per batch, average reward per batch, and number of data in each batch.
    '''
    def run_epoch(self, epoch_data, epoch_gt, batch_size, phase):
        correct_count = list()
        rewards_list = list()
        batch_count = list()
        preds_list = list()
        probs_list = list()
        for batch_head in range(0, epoch_data.shape[0], batch_size):
            batch_end = batch_head + batch_size
            batch_data = epoch_data[batch_head:batch_end]
            if epoch_gt is None:
                batch_gt = None
            else:
                batch_gt = epoch_gt[batch_head:batch_end]
            batch_correct_count, batch_reward, batch_preds, batch_probs = \
                self.run_batch(batch_data, batch_gt, phase)
            correct_count.append(batch_correct_count)
            rewards_list.append(batch_reward)
            batch_count.append(batch_data.shape[0])
            preds_list.append(batch_preds)
            probs_list.append(batch_probs)
            self.reset()
        return correct_count, rewards_list, batch_count, preds_list, probs_list
    
    def save(self, fname):
        os.system('mkdir -p snapshots/fname')
        self.actor_net.save('snapshots/fname/'+fname+'_actor_net.h5')
        self.action_net.save('snapshots/fname/'+fname+'_action_net.h5')
        self.actor_state.save('snapshots/fname/'+fname+'_actor_state')
        self.action_state.save('snapshots/fname/'+fname+'_action_state')
        np.save('snapshots/fname/'+fname+'_value_weights', self.value_weights)
        
    def load(self, fname):
        self.actor_net.load('snapshots/fname/'+fname+'_actor_net.h5')
        self.action_net.load('snapshots/fname/'+fname+'_action_net.h5')
        self.actor_state.load('snapshots/fname/'+fname+'_actor_state')
        self.action_state.load('snapshots/fname/'+fname+'_action_state')
        self.value_weights = np.load('snapshots/fname/'+fname+'_value_weights')
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    