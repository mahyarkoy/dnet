#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:19:39 2017

@author: mahyar
"""

import numpy as np
import apollocaffe
from apollocaffe.layers import *
import dnet
import matplotlib.pyplot as plt
from sklearn import svm
import os

log_path = 'log'

apollocaffe.set_random_seed(0)
np.random.seed(0)
apollocaffe.set_device(0)

os.system('mkdir -p '+log_path+'/reward_plots')

def train_dnet(input_data, input_labels, epochs, fignum=-1):
    batch_size = 10
    save_period = 5
    log_path = 'log'
    ### initialize decision net class
    dec_net = dnet.Dnet()
    order = np.arange(input_data.shape[0])
    ep_rewards = list()
    for ep in range(epochs):
        print '>>> At epoch %d' % ep
        ### shuffle data and labels
        np.random.shuffle(order)
        copy_data = input_data[order,:]
        copy_labels = input_labels[order]

        ### train for one epoch
        correct_count, average_reward, batch_count, _, _ = \
            dec_net.run_epoch(copy_data, copy_labels, batch_size, phase='train')
        
        ep_rewards += average_reward
        ### print results
        acc = np.sum(correct_count) * 100.0 / np.sum(batch_count)
        reward = np.mean(average_reward)
        result_summary = '>>> epoch %d >>> acc=%4.3f reward=%4.3f' % (ep, acc, reward)
        print result_summary
        with open(log_path+'/dnet_summary.txt', 'a+') as rf:
            print >>rf, result_summary
            
        ### plot rewards progress
        #plt.figure(fignum, figsize=(8,6))
        #plt.plot(np.array(average_reward))
        #plt.title('Average Reward at epoch %d' % ep)
        ##ax = plt.gca()
        ##start, end = ax.get_xlim()
        ##ax.xaxis.set_ticks(np.arange(start, end, 1.0))
        #plt.grid(True, which='both', linestyle='dotted')
        #plt.savefig(log_path + '/reward_plots/plot_%d' % ep + '.pdf')
        #if ep % save_period == 0 and ep != 0:
        #    dec_net.save('dnet_%d' % ep)
    
    ### plot rewards progress over all batches
    plt.figure(fignum, figsize=(8,6))
    window_size = 10*batch_size
    plt.plot(np.convolve(np.array(ep_rewards), np.ones(window_size), mode='valid') * 1.0/window_size)
    plt.title('Average Reward at epoch %d' % ep)
    #ax = plt.gca()
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start, end, 1.0))
    plt.grid(True, which='both', linestyle='dotted')
    plt.legend([str(num) for num in range(epochs)], loc='best')
    
    X = input_data
    Y = input_labels
    if fignum > -1:
        ### draw decision boundaries
        plt.figure(fignum+1, figsize=(8, 6))
        #plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=9)#, cmap=plt.cm.Paired)
        #plt.axis('tight')
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        
        _, _, _, preds_list, probs_list = \
            dec_net.run_epoch(np.c_[XX.ravel(), YY.ravel()], None, batch_size, phase='test')
        preds = np.concatenate(preds_list)
        probs = np.concatenate(probs_list)
        preds = preds.reshape(XX.shape)
        probs = probs.reshape(XX.shape)
        ### hacky way to make pos and neg score based on probs
        #mask = (preds > 0).astype(float)
        #negprobs = mask * probs + (mask-1) * probs
        plt.figure(fignum+1, figsize=(8, 6))
        plt.pcolormesh(XX, YY, preds, cmap=plt.cm.Paired)
        #plt.contour(XX, YY, negprobs, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
        #            levels=[-0.5, 0., 0.5])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('DNet decision boundaries')
    
        ### action preference
        plt.figure(fignum+2, figsize=(8, 6))
        action_preds = \
            dec_net.actor_forward(np.c_[XX.ravel(), YY.ravel()], greedy=False)
        actions = np.array([int(a) for a in action_preds])
        actions = actions.reshape(XX.shape)
        plt.pcolormesh(XX, YY, actions, cmap=plt.cm.Accent)
        plt.title('Subnet selection in each region')
        
    return dec_net

def train_svm(input_data, input_labels, kernel, title='SVM', fignum=1):
    X = input_data
    Y = input_labels
    clf = svm.SVC(kernel=kernel, gamma=100.0)
    clf.fit(X, Y)
    acc = clf.score(X, Y)
    print '>>> SVM accuracy=%4.3f' % acc
    
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(8, 6))
    #plt.clf()

    #plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
    #            facecolor='none', edgecolor='black', zorder=10)
    #plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=9, cmap=plt.cm.Paired)
    
    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(8, 6))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    #plt.xticks(())
    #plt.yticks(())
     
    return clf

def generate_normal_data(train_size, test_size, centers, stds, labels):
    num_class = len(centers)
    num_dim = len(centers[0])
    train_class_size = train_size / num_class
    test_class_size = test_size / num_class
    ### initialize datasets and labels
    train_dataset = np.empty((train_size, num_dim))
    train_gt = np.empty(train_size)
    test_dataset = np.empty((test_size, num_dim))
    test_gt = np.empty(test_size)
    n = 0
    for c, v, l in zip(centers, stds, labels):
        train_data = np.random.multivariate_normal(c, v*np.eye(len(c)), size=train_class_size)
        train_labels = l*np.ones(train_class_size)
        test_data = np.random.multivariate_normal(c, v*np.eye(len(c)), size=test_class_size)
        test_labels = l*np.ones(test_class_size)
        ### storing into datasets and labels
        train_dataset[n*train_class_size:(n+1)*train_class_size, ...] = train_data
        train_gt[n*train_class_size:(n+1)*train_class_size] = train_labels
        test_dataset[n*test_class_size:(n+1)*test_class_size, ...] = test_data
        test_gt[n*test_class_size:(n+1)*test_class_size] = test_labels
        n += 1
    ### shuffle data
    order = np.arange(train_size)
    np.random.shuffle(order)
    train_dataset = train_dataset[order, ...]
    train_gt = train_gt[order].astype(int)
    return train_dataset, train_gt, test_dataset, test_gt

def generate_struct_data(data_size, lb, ub, std):
    x = np.random.uniform(lb, ub, data_size)
    yt = 2*np.sin(x*10)
    y = np.random.normal(yt, 1, size=yt.size)
    labels = [1 if l > lt else 0 for l, lt in zip(y,yt)]
    dataset = np.c_[x,y]
    return dataset, np.array(labels)

def plot_dataset(dataset, gt, title='Dataset'):
    ### plot the dataset
    plt.figure(1,figsize=(8,6))
    plt.scatter(dataset[:,0], dataset[:,1], c=gt.astype(int))
    plt.title(title)
    ax = plt.gca()
    start, end = ax.get_xlim()
    start = np.floor(start)
    end = np.ceil(end)
    ax.xaxis.set_ticks(np.arange(start, end, 1.0))
    plt.grid(True, which='both', linestyle='dotted')
    #plt.savefig(log_path + 'train_dataset_plot' + '.pdf')
    return ax

if __name__ == '__main__':
    #centers = [[2,0], [0,2], [-2,0], [0,-2]]
    #stds = [0.5, 0.5, 0.5, 0.5]
    #labels = [0, 1, 0, 1]
    
    #centers = [[1,1], [2,2], [-1,1], [-2,2], [-1,-1], [-2,-2], [1,-1], [2,-2]]
    #stds = [0.25] * 8
    #labels = [0, 1, 1, 0, 0, 1, 1, 0]
    train_size = 1000
    test_size = 100
    #train_dataset, train_gt, test_dataset, test_gt = \
    #    generate_normal_data(train_size, test_size, centers, stds, labels)
    train_dataset, train_gt = generate_struct_data(train_size, -2.0, 2.0, 1)
    #train_dataset -= np.mean(train_dataset, axis=0)
    #train_dataset /= np.std(train_dataset, axis=0)
    plot_dataset(train_dataset, train_gt, 'XOR Dataset')
    psvm = train_svm(train_dataset, train_gt, 'rbf', title='svm', fignum=2)
    pdnet = train_dnet(train_dataset, train_gt, 50, fignum=5)
        
    
        
        
    
    