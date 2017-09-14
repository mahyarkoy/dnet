#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:10:34 2017

@author: mahyar
"""
### To convert to mp4 in command line
# ffmpeg -framerate 25 -i fields/field_%d.png -c:v libx264 -pix_fmt yuv420p baby_log_15.mp4
### To speed up mp4
# ffmpeg -i baby_log_57.mp4 -r 100 -filter:v "setpts=0.1*PTS" baby_log_57_100.mp4
# for i in {0..7}; do mv baby_log_a"$((i))" baby_log_"$((i+74))"; done

import numpy as np
import baby_gan
import matplotlib.pyplot as plt
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import matplotlib.tri as mtri
from sklearn.neighbors.kde import KernelDensity
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
args = arg_parser.parse_args()
log_path = args.log_path
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
    if gt is None:
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

def baby_gan_field_2d(baby, x_min, x_max, y_min, y_max, batch_size):
    XX, YY = np.mgrid[x_min:x_max:80j, y_min:y_max:80j]
    data_mat = np.c_[XX.ravel(), YY.ravel()]
    logits = np.zeros(data_mat.shape[0])
    for batch_start in range(0, data_mat.shape[0], batch_size):
        batch_end = batch_start + batch_size
        batch_data = data_mat[batch_start:batch_end, ...]
        logits[batch_start:batch_end] = baby.step(batch_data, batch_size, dis_only=True)
    Z = logits.reshape(XX.shape)
    tri = mtri.Triangulation(XX.flatten(), YY.flatten())
    return (XX, YY, Z, (x_min, x_max, y_min, y_max), tri)

def baby_gan_field_1d(baby, x_min, x_max, batch_size):
    XX = np.linspace(x_min, x_max, 80)
    data_mat = XX.reshape((XX.size,1))
    logits = np.zeros(data_mat.shape[0])
    for batch_start in range(0, data_mat.shape[0], batch_size):
        batch_end = batch_start + batch_size
        batch_data = data_mat[batch_start:batch_end,:]
        logits[batch_start:batch_end] = baby.step(batch_data, batch_size, dis_only=True)
    return (data_mat, logits, (x_min, x_max))


def plot_field_2d(field_params, (r_data, br_data), (g_data, bg_data), fignum, save_path, title):
    # plot the line, the points, and the nearest vectors to the plane
    fig = plt.figure(fignum, figsize=(12,20))
    fig.clf()
    ### top subplot: decision boundary
    ax = fig.add_subplot(2, 1, 1)
    ax.scatter(br_data[:, 0], br_data[:, 1], c='r', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
    ax.scatter(bg_data[:, 0], bg_data[:, 1], c='b', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
    
    probs = 1.0 / (1.0 + np.exp(-field_params[2]))
    #z_min = 0.0
    #z_max = 1.0
    z_min = probs.min()
    z_max = probs.max()
    dec = ax.pcolor(field_params[0], field_params[1], probs, cmap='coolwarm', vmin=z_min, vmax=z_max)
    ax.axis([field_params[3][0], field_params[3][1], field_params[3][2], field_params[3][3]])
    fig.colorbar(dec, shrink=0.5, aspect=10)
    ax.contour(field_params[0], field_params[1], field_params[2], colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-1.0, 0.0, 1.0])
    ax.set_title(title+'_sig_boundary')

    ### second subplot: logit score surface
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    #surf = ax.plot_trisurf(field_params[0].flatten(), field_params[1].flatten(), field_params[2].flatten(),
    #    triangles=field_params[4].triangles, cmap=cm.CMRmap, alpha=0.2)
    surf = ax.plot_surface(field_params[0], field_params[1], field_params[2], rstride=1, cstride=1, cmap=cm.CMRmap,
                       linewidth=1, antialiased=False, alpha=0.6)
    #ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    cset = ax.contour(field_params[0], field_params[1], field_params[2], zdir='x', offset=-5, cmap=cm.Spectral)
    cset = ax.contour(field_params[0], field_params[1], field_params[2], zdir='y', offset=5, cmap=cm.Spectral)
    ax.set_title(title+'_score_surf')
    
    fig.savefig(save_path)

def plot_field_1d(field_params, (r_data, br_data), (g_data, bg_data), fignum, save_path, title):
    ### Estimate densities for real and generated data
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(r_data)
    r_dens = np.exp(kde.score_samples(field_params[0]))
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(g_data)
    g_dens = np.exp(kde.score_samples(field_params[0]))

    ### plot the line, the points, and the nearest vectors to the plane
    fig = plt.figure(fignum, figsize=(12,20))
    fig.clf()

    ### top subplot: decision boundary and prob densities
    ax = fig.add_subplot(2, 1, 1)
    y_data = np.zeros(br_data.shape)
    ax.scatter(br_data, y_data, c='r', zorder=9, edgecolor='black', marker='+')
    ax.scatter(bg_data, y_data, c='b', zorder=9, edgecolor='black', marker='+')
    
    probs = 1.0 / (1.0 + np.exp(-field_params[1]))
    #z_min = 0.0
    #z_max = 1.0
    z_min = probs.min()
    z_max = probs.max()
    rl, = ax.plot(field_params[0], r_dens, 'r')
    gl, = ax.plot(field_params[0], g_dens, 'b')
    pl, = ax.plot(field_params[0], probs, 'g')
    ax.legend((rl, gl, pl), ('Real', 'Gen', 'Sig'), loc=2)
    ax.grid(True, which='both', linestyle='dotted')
    ax.set_xlabel('Data Space')
    ax.set_ylabel('Prob Density')
    ax.set_title(title+'_sig_boundary')
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-5.1, 5.1)

    ### bottom subplot: decision function DIS
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(field_params[0], field_params[1], 'm')
    ax.grid(True, which='both', linestyle='dotted')
    ax.set_xlabel('Data Space')
    ax.set_ylabel('DIS Value')
    ax.set_title(title+'_score_surf')

    fig.savefig(save_path)

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear'):
    plt.figure(fignum, figsize=(8, 6))
    plt.clf()

    plt.plot(vals, color=color)
    plt.grid(True, which='both', linestyle='dotted')
    plt.title(name)
    plt.xlabel('Iterations')
    plt.ylabel('Values')
    if ytype=='log':
        plt.yscale('log')
    plt.savefig(save_path)

def plot_time_mat(mat, mat_names, fignum, save_path):
    for n in range(mat.shape[1]):
        fig_name = mat_names[n]
        ytype = 'log' if 'param' in fig_name else 'linear'
        plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype)

if __name__ == '__main__':
    ### dataset definition
    train_size = 51200
    test_size = 100
    data_dim = 2
    fov = 4 ## field of view in field plot

    centers = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    stds = [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
    labels = [0, 0, 0, 0]
    train_dataset, train_gt, test_dataset, test_gt = \
        generate_normal_data(train_size, test_size, centers, stds, labels)
    plot_dataset(train_dataset, train_gt, 'XOR Dataset')

    ### normalize input dataset
    #train_mu = np.mean(train_dataset, axis=0)
    #train_std = np.std(train_dataset, axis=0)
    #train_dataset = (train_dataset - train_mu) / train_std

    ### logs initi
    g_logs = list()
    d_r_logs = list()
    d_g_logs = list()

    ### baby gan training
    epochs = 100
    d_updates = 10
    g_updates = 1
    baby = baby_gan.BabyGAN(data_dim)
    batch_size = 512
    field_sample_size = 100
    itr = 0
    itr_total = 0
    g_itr = 0
    d_itr = 0
    g_max_itr = 1e4
    max_itr_total = np.ceil(train_size*1.0 / batch_size + train_size*1.0 / batch_size / d_updates * g_updates)
    widgets = ["baby_gan", Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=g_max_itr, widgets=widgets)
    pbar.start()
    while g_itr < g_max_itr:
        #np.random.shuffle(train_dataset)
        train_dataset, train_gt, test_dataset, test_gt = \
            generate_normal_data(train_size, test_size, centers, stds, labels)
        #print '>>> Epoch %d is started...' % ep
        ### discriminator update
        for batch_start in range(0, train_size, batch_size):
            pbar.update(g_itr)
            batch_end = batch_start + batch_size
            batch_data = train_dataset[batch_start:batch_end, 0] if data_dim == 1 else train_dataset[batch_start:batch_end, :]
            batch_data = batch_data.reshape((batch_data.shape[0], data_dim))
            logs, batch_g_data = baby.step(batch_data, batch_size, gen_update=False)
            if data_dim == 1:
                g_data = baby.step(None, field_sample_size, gen_only=True)
                d_data = train_dataset[0:field_sample_size, 0]
                d_data = d_data.reshape((d_data.shape[0], data_dim))
            else:
                g_data = baby.step(None, field_sample_size, gen_only=True)
                d_data = train_dataset[0:field_sample_size, :]
                d_data = d_data.reshape((d_data.shape[0], data_dim))
            g_logs.append(logs[0])
            d_r_logs.append(logs[1])
            d_g_logs.append(logs[2])
            ### calculate and plot field of decision
            if data_dim == 1:
                field_params = baby_gan_field_1d(baby, -fov, fov, batch_size*10)
                plot_field_1d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
                    log_path_png+'/field_%d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))    
            else:
                field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
                plot_field_2d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
                    log_path_png+'/field_%d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))
            d_itr += 1
            itr_total += 1
            ### generator updates: g_updates times for each d_updates of discriminator
            if d_itr % d_updates == 0 and d_itr != 0:
                for gn in range(g_updates):
                    logs, batch_g_data = baby.step(batch_data, batch_size, gen_update=True)
                    g_data = baby.step(None, field_sample_size, gen_only=True)
                    g_logs.append(logs[0])
                    d_r_logs.append(logs[1])
                    d_g_logs.append(logs[2])
                    if data_dim == 1:
                        plot_field_1d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
                            log_path_png+'/field_%d.png' % itr_total, 'GEN_%d_%d_%d' % (gn, g_itr, itr_total))
                    else:
                        plot_field_2d(field_params, (d_data, batch_data), (g_data, batch_g_data), 0,
                            log_path_png+'/field_%d.png' % itr_total, 'GEN_%d_%d_%d' % (gn, g_itr, itr_total))
                    g_itr += 1
                    itr_total += 1
                #_, dis_confs, trace = baby.gen_consolidate(count=50)
                #print '>>> CONFS: ', dis_confs
                #print '>>> TRACE: ', trace
                #baby.reset_network('d_')

    ### plot baby gan progress logs
    g_logs_mat = np.array(g_logs)
    d_r_logs_mat = np.array(d_r_logs)
    d_g_logs_mat = np.array(d_g_logs)
    g_logs_names = ['g_d_acc', 'g_loss', 'g_logit_data', 'g_logit_diff', 'g_out_diff', 'g_param_diff']
    d_r_logs_names = ['d_r_acc', 'd_r_loss', 'd_r_logit_data', 'd_r_logit_diff', 'd_r_param_diff']
    d_g_logs_names = ['d_g_acc', 'd_g_loss', 'd_g_logit_data', 'd_g_logit_diff', 'd_g_param_diff']

    plot_time_mat(g_logs_mat, g_logs_names, 1, log_path)
    plot_time_mat(d_r_logs_mat, d_r_logs_names, 1, log_path)
    plot_time_mat(d_g_logs_mat, d_g_logs_names, 1, log_path)





        
    
        
        
    
    




















