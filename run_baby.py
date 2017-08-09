#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:10:34 2017

@author: mahyar
"""

import numpy as np
import baby_gan
import matplotlib.pyplot as plt
import os
from progressbar import ETA, Bar, Percentage, ProgressBar

log_path = 'baby_log'
log_path_png = log_path+'/fields'
os.system('mkdir -p '+log_path_png)

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
        train_data = np.random.multivariate_normal(c, np.diag(v), size=train_class_size)
        train_labels = l*np.ones(train_class_size)
        test_data = np.random.multivariate_normal(c, np.diag(v), size=test_class_size)
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
    if gt == None:
        gt = np.ones(dataset.shape[0])
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

def baby_gan_field(baby, x_min, x_max, y_min, y_max, batch_size):
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    data_mat = np.c_[XX.ravel(), YY.ravel()]
    logits = np.zeros(data_mat.shape[0])
    for batch_start in range(0, data_mat.shape[0], batch_size):
        batch_end = batch_start + batch_size
        batch_data = data_mat[batch_start:batch_end, ...]
        logits[batch_start:batch_end] = baby.step(batch_data, batch_size, dis_only=True)
    Z = logits.reshape(XX.shape)
    return (XX, YY, Z, (x_min, x_max, y_min, y_max))

def plot_field(field_params, r_data, g_data, fignum, save_path, title):
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(10, 10))
    plt.clf()

    plt.scatter(r_data[:, 0], r_data[:, 1], c='r', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
    plt.scatter(g_data[:, 0], g_data[:, 1], c='b', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
    
    '''
    plt.axis('tight')
    plt.figure(fignum, figsize=(8, 6))
    plt.pcolormesh(field_params[0], field_params[1], field_params[2] > 0, cmap=plt.cm.Paired)
    plt.contour(field_params[0], field_params[1], field_params[2], colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(field_params[3][0], field_params[3][1])
    plt.ylim(field_params[3][2], field_params[3][3])
    '''
    probs = 1.0 / (1.0 + np.exp(-field_params[2]))
    z_min = 0.0
    z_max = 1.0
    plt.pcolor(field_params[0], field_params[1], probs, cmap='coolwarm', vmin=z_min, vmax=z_max)
    plt.axis([field_params[3][0], field_params[3][1], field_params[3][2], field_params[3][3]])
    #plt.colorbar()
    plt.contour(field_params[0], field_params[1], probs, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[.4, 0.5, .6])
    plt.title(title)

    plt.savefig(save_path)

def plot_time_series(name, vals, fignum, save_path, color='b'):
    plt.figure(fignum, figsize=(8, 6))
    plt.clf()

    plt.plot(vals, color=color)
    plt.grid(True, which='both', linestyle='dotted')
    plt.title(name)
    plt.xlabel('Iterations')
    plt.ylabel('Values')

    plt.savefig(save_path)

def plot_time_mat(mat, mat_names, fignum, save_path):
    for n in range(mat.shape[1]):
        fig_name = mat_names[n]
        plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.pdf')

if __name__ == '__main__':
    ### dataset definition
    train_size = 6400
    test_size = 100

    centers = [[5,5], [-5,-5]]
    stds = [[0.2, 0.2]] * 2
    labels = [0, 0]
    train_dataset, train_gt, test_dataset, test_gt = \
        generate_normal_data(train_size, test_size, centers, stds, labels)
    plot_dataset(train_dataset, train_gt, 'XOR Dataset')

    ### logs initi
    g_logs = list()
    d_r_logs = list()
    d_g_logs = list()

    ### baby gan training
    epochs = 10
    d_updates = 50
    g_updates = 50
    baby = baby_gan.BabyGAN()
    batch_size = 16
    itr = 0
    itr_total = 0
    max_itr_total = np.ceil(train_size*1.0 / batch_size + train_size*1.0 / batch_size / d_updates * g_updates)
    np.random.shuffle(train_dataset)
    widgets = ["baby_gan", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=max_itr_total*epochs, widgets=widgets)
    pbar.start()
    for ep in range(epochs):
        print '>>> Epoch %d is started...' % ep
        ### discriminator update
        for batch_start in range(0, train_size, batch_size):
            pbar.update(itr_total)
            batch_end = batch_start + batch_size
            batch_data = train_dataset[batch_start:batch_end, ...]
            logs, g_data = baby.step(batch_data, batch_size, gen_update=False, dis_only=False)
            g_logs.append(logs[0])
            d_r_logs.append(logs[1])
            d_g_logs.append(logs[2])
            ### calculate and plot field of decision
            field_params = baby_gan_field(baby, -20., 20., -20., 20., batch_size*10)
            plot_field(field_params, batch_data, g_data, 0, log_path_png+'/field_%d.png' % itr_total, 'dis_update')
            itr += 1
            itr_total += 1
            ### generator updates: g_updates times for each d_updates of discriminator
            if itr % d_updates == 0 and itr != 0:
                for gn in range(g_updates):
                    logs, g_data = baby.step(batch_data, batch_size, gen_update=True, dis_only=False)
                    g_logs.append(logs[0])
                    d_r_logs.append(logs[1])
                    d_g_logs.append(logs[2])
                    plot_field(field_params, batch_data, g_data, 0, log_path_png+'/field_%d.png' % itr_total, 'gen_update')
                    itr_total += 1

    ### plot baby gan progress logs
    g_logs_mat = np.array(g_logs)
    d_r_logs_mat = np.array(d_r_logs)
    d_g_logs_mat = np.array(d_g_logs)
    g_logs_names = ['d_g_acc', 'd_loss', 'g_logit_diff', 'g_out_diff', 'g_param_diff']
    d_r_logs_names = ['d_r_acc', 'd_r_loss', 'd_r_logit_data', 'd_r_logit_diff', 'd_r_param_diff']
    d_g_logs_names = ['d_g_acc', 'd_g_loss', 'd_g_logit_data', 'd_g_logit_diff', 'd_g_param_diff']

    plot_time_mat(g_logs_mat, g_logs_names, 1, log_path)
    plot_time_mat(d_r_logs_mat, d_r_logs_names, 1, log_path)
    plot_time_mat(d_g_logs_mat, d_g_logs_names, 1, log_path)





        
    
        
        
    
    




















