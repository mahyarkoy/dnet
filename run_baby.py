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
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

#import baby_gan
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm
import matplotlib.tri as mtri
from sklearn.neighbors.kde import KernelDensity
import argparse
print matplotlib.get_backend()
import scipy.stats as sc_stats
import matplotlib.cm as mat_cm
import cPickle as pk

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
arg_parser.add_argument('-e', '--eval', dest='eval_int', required=True, help='eval intervals.')
arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
args = arg_parser.parse_args()
log_path = args.log_path
eval_int = int(args.eval_int)
run_seed = int(args.seed)

np.random.seed(run_seed)
tf.set_random_seed(run_seed)

import tf_baby_gan
import vee_gan

log_path_png = log_path+'/fields'
log_path_snap = log_path+'/snapshots'
log_path_manifold = log_path+'/manifolds'
log_path_data = log_path+'/data_img'
os.system('mkdir -p '+log_path_png)
os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+log_path_manifold)
os.system('mkdir -p '+log_path_data)

def generate_normal_data(sample_size, centers, stds, sample_ratio=None, labels=None):
	### TODO: handle excessive samples randomly
	num_class = len(centers)
	num_dim = len(centers[0])
	### initialize datasets and labels
	dataset = np.empty((sample_size, num_dim))
	gt = np.empty(sample_size)
	if labels is None:
		labels = np.zeros(num_class)
	if sample_ratio is None:
		sample_ratio = np.ones(num_class)
	sample_ratio_array = 1.0 * np.array(sample_ratio) / np.sum(sample_ratio)
	samples_per_class = np.floor(sample_ratio_array * sample_size)
	samples_per_class[-1] += sample_size - np.sum(samples_per_class)
	start = 0
	for c, v, s, l in zip(centers, stds, samples_per_class, labels):
		class_size = int(s)
		data = np.random.multivariate_normal(c, np.diag(v), size=class_size)
		labels = l*np.ones(class_size)
		### storing into datasets and labels
		dataset[start:start+class_size, ...] = data
		gt[start:start+class_size] = labels
		start += class_size
	### shuffle data
	order = np.arange(sample_size)
	np.random.shuffle(order)
	dataset = dataset[order, ...]
	gt = gt[order].astype(int)
	return dataset, gt

def generate_struct_data(data_size, lb, ub, std):
	x = np.random.uniform(lb, ub, data_size)
	yt = 2*np.sin(x*10)
	y = np.random.normal(yt, 1, size=yt.size)
	labels = [1 if l > lt else 0 for l, lt in zip(y,yt)]
	dataset = np.c_[x,y]
	return dataset, np.array(labels)

def generate_circle_data(data_size):
	num_comp = 2
	z = np.random.uniform(0.0, 2*np.pi, data_size)
	ch = np.random.choice(num_comp, size=data_size, replace=True, p=[0.5, 0.5])
	
	x1 = np.sin(z) - 2
	y1 = np.cos(z)

	x2 = np.sin(z) + 2
	y2 = np.cos(z)

	dx = np.c_[x1, x2]
	dy = np.c_[y1, y2]

	data = np.c_[dx[np.arange(data_size), ch], dy[np.arange(data_size), ch]]
	return data

def generate_line_data(data_size):
	num_lines = 4
	lb = 0.
	ub = 1.
	z = np.random.uniform(lb, ub, data_size)
	mu, sig = 0.5, 0.2
	z_n = sc_stats.truncnorm((lb-mu)/sig, (ub-mu)/sig, loc=mu, scale=sig).rvs(data_size)
	#ch = np.random.randint(0, num_lines, data_size)
	ch = np.random.choice(num_lines, size=data_size, replace=True, p=[0.25, 0.25, 0.25, 0.25])
	
	x1 = z * .25 + (1-z) * .75
	x1_n = z_n * .25 + (1-z_n) * .75
	y1 = -1. * x1 + 1.
	
	x2 = -x1 
	y2 = -1. * x2 - 1.

	x3 = x1
	y3 = 1. * x3 - 1.

	x4 = -x1
	y4 = 1. * x4 + 1.

	dx = np.c_[x1, x2, x3, x4]
	dy = np.c_[y1, y2, y3, y4]
	data = np.c_[dx[np.arange(data_size), ch], dy[np.arange(data_size), ch]]
	#data = np.c_[x1, y1]
	return data

def generate_dot_data(data_size):
	data = np.random.choice([2., -2.], size=data_size, replace=True, p=[0.5, 0.5])
	return data

def plot_dataset(datasets, color, pathname, title='Dataset', fov=2):
	### plot the dataset
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	#plt.scatter(dataset[:,0], dataset[:,1], c=gt.astype(int))

	for i, d in enumerate(datasets):
		d = d.reshape([d.size, 1]) if len(d.shape) == 1 else d
		if d.shape[-1] == 1:
			d = np.c_[d, np.ones(d.shape)]
		ax.scatter(d[:,0], d[:,1], c=color[i], marker='.', edgecolors='none')
	ax.set_title(title)
	ax.set_xlim(-fov, fov)
	ax.set_ylim(-fov, fov)
	ax.grid(True, which='both', linestyle='dotted')
	fig.savefig(pathname, dpi=300)
	plt.close(fig)

def plot_dataset_gid(baby, data_size, color_map, pathname, fov=2):
	g_num = baby.g_num
	cmap = mat_cm.get_cmap(color_map)
	rgb_colors = cmap(1.0 * np.arange(baby.g_num) / baby.g_num)
	rgb_colors[:,3] = 0.2
	g_data = list()
	for i in range(g_num):
		z = i * np.ones(data_size)
		g_data.append(sample_baby_gan(baby, data_size, z_data=z))
		g_color = np.array(rgb_colors[i,:].reshape(1, 4))
		g_color[:,3] = 1.
		plot_dataset([g_data[-1]], g_color, pathname+'_gid_'+str(i)+'.png', 'gid_'+str(i), fov=fov)
	plot_dataset(g_data, rgb_colors.reshape(-1, 4), pathname+'_gids.png', 'Generators', fov=fov)

def plot_dataset_en(baby, dataset, color_map, pathname, title='Dataset', fov=2, color_bar=True):
	### plot the dataset
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()

	d = dataset.reshape([dataset.size, 1]) if len(dataset.shape) == 1 else dataset
	en_logits = eval_baby_en(baby, d)
	cid = np.argmax(en_logits, axis=1)
	if d.shape[-1] == 1:
		d = np.c_[d, np.ones(d.shape)]
	dec = ax.scatter(d[:,0], d[:,1], c=cid, cmap=color_map, 
		marker='.', edgecolors='none', vmin=0, vmax=baby.g_num-1)

	if color_bar is True:
		fig.colorbar(dec)
	ax.set_title(title)
	ax.set_xlim(-fov, fov)
	ax.set_ylim(-fov, fov)
	ax.grid(True, which='both', linestyle='dotted')
	fig.savefig(pathname, dpi=300)
	plt.close(fig)

def plot_manifold_1d(baby, pathname, title='Generator Function'):
	data_size = 200
	z_range = baby.z_range
	zi = np.linspace(-z_range, z_range, data_size)
	data = sample_baby_gan(baby, data_size, zi_data=zi.reshape(data_size, 1))

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	ax.plot(zi, data, 'b')
	ax.set_title(title)
	ax.set_xlim(-1.5, 1.5)
	ax.set_ylim(-4, 4)
	ax.grid(True, which='both', linestyle='dotted')
	fig.savefig(pathname, dpi=300)
	plt.close(fig)
	
'''
Plots generator 2D output data versus hidden z
'''
def plot_manifold(baby, batch_size, fignum, save_path, title, fov=1.0):
	z_range = baby.z_range
	ZZ = np.linspace(-z_range, z_range, 200)
	XX = np.zeros(ZZ.shape[0])
	YY = np.zeros(ZZ.shape[0])
	### calculate the generator data manifold
	for batch_start in range(0, ZZ.shape[0], batch_size):
		batch_end = batch_start + batch_size
		batch_data = ZZ[batch_start:batch_end, ...]
		gen_data = baby.step(None, batch_size, gen_only=True, 
			z_data=batch_data.reshape(batch_data.shape[0], 1))
		XX[batch_start:batch_end] = gen_data[:,0]
		YY[batch_start:batch_end] = gen_data[:,1]

	#XX = np.sin(2*np.pi*ZZ)
	#YY = np.cos(2*np.pi*ZZ)
	### plot the 3d manifold
	fig = plt.figure(fignum, figsize=(8, 6))
	fig.clf()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	ax.plot(XX, YY, ZZ, zdir='z', c='b', linewidth=3)
	ax.set_xlim(-fov, fov)
	ax.set_ylim(-fov, fov)
	ax.set_zlim(-z_range, z_range)
	### plot the 2d projections X and Y on Z
	ax.plot(XX, np.zeros(XX.shape) + fov, ZZ, zdir='z', c='m')
	ax.plot(np.zeros(XX.shape) - fov, YY, ZZ, zdir='z', c='g')
	ax.plot(XX, YY, np.zeros(XX.shape) - z_range, zdir='z', c='r')
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title(title+'_generator_manifold')	
	fig.savefig(save_path, dpi=300)

def baby_gan_field_2d(baby, x_min, x_max, y_min, y_max, batch_size=128):
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


def plot_field_2d(field_params, fov, fignum, save_path, title, 
	r_data=None, br_data=None, g_data=None, bg_data=None):
	# plot the line, the points, and the nearest vectors to the plane
	fig = plt.figure(0, figsize=(12,20))
	fig.clf()
	### top subplot: decision boundary
	ax = fig.add_subplot(2, 1, 1)
	if r_data is not None:
		ax.scatter(r_data[:, 0], r_data[:, 1], c='#FF8B74', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
	if br_data is not None:
		ax.scatter(br_data[:, 0], br_data[:, 1], c='r', zorder=10, cmap=plt.cm.Paired, edgecolor='black')
	if g_data is not None:
		ax.scatter(g_data[:, 0], g_data[:, 1], c='#74CFFF', zorder=9, cmap=plt.cm.Paired, edgecolor='black')
	if bg_data is not None:
		ax.scatter(bg_data[:, 0], bg_data[:, 1], c='b', zorder=10, cmap=plt.cm.Paired, edgecolor='black')

	
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
	surf = ax.plot_surface(field_params[0], field_params[1], field_params[2], rstride=1, cstride=1, cmap=cm.jet,
					   linewidth=0, antialiased=False, alpha=0.7, edgecolors='lightgrey')
	#ax.set_zlim(field_params[2].min()-1, field_params[2].max())
	zlim_low = min(field_params[2].min()-1, -5)
	zlim_high = max(field_params[2].max(), 5)
	ax.set_zlim(zlim_low, zlim_high)
	fig.colorbar(surf, shrink=0.5, aspect=10)
	#cset = ax.contour(field_params[0], field_params[1], field_params[2], zdir='x', offset=field_params[3][0]-0.5, cmap=cm.jet)
	#cset = ax.contour(field_params[0], field_params[1], field_params[2], zdir='y', offset=field_params[3][3]+0.5, cmap=cm.jet)
	cset = ax.contour(field_params[0], field_params[1], field_params[2], zdir='z', offset=zlim_low, cmap=cm.jet)
	ax.set_title(title+'_score_surf')
	
	fig.savefig(save_path, dpi=300)

def plot_field_1d(field_params, r_data, br_data, g_data, bg_data, fignum, save_path, title):
	### Estimate densities for real and generated data
	kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(r_data)
	r_dens = np.exp(kde.score_samples(field_params[0]))
	kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(g_data)
	g_dens = np.exp(kde.score_samples(field_params[0]))

	### plot the line, the points, and the nearest vectors to the plane
	fig = plt.figure(fignum, figsize=(12,20))
	fig.clf()

	### top subplot: decision boundary and prob densities
	plot_r_data = r_data
	plot_g_data = g_data
	ax = fig.add_subplot(2, 1, 1)
	y_data = np.zeros(plot_r_data.shape)
	ax.scatter(plot_r_data, y_data, c='r', zorder=9, edgecolor='black', marker='+')
	ax.scatter(plot_g_data, y_data, c='b', zorder=9, edgecolor='black', marker='+')
	
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

	fig.savefig(save_path, dpi=300)

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear', itrs=None):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	if itrs is None:
		ax.plot(vals, color=color)	
	else:
		ax.plot(itrs, vals, color=color)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title(name)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Values')
	if ytype=='log':
		ax.set_yscale('log')
	fig.savefig(save_path, dpi=300)
	plt.close(fig)

def plot_time_mat(mat, mat_names, fignum, save_path, ytype=None, itrs=None):
	for n in range(mat.shape[1]):
		fig_name = mat_names[n]
		if not ytype:
			ytype = 'log' if 'param' in fig_name else 'linear'
		plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype, itrs=itrs)

'''
Sample sample_size data points from baby.
'''
def sample_baby_gan(baby, sample_size, batch_size=512, z_data=None, zi_data=None):
	g_samples = np.zeros([sample_size, baby.data_dim])
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_zi = zi_data[batch_start:batch_end, ...] if zi_data is not None else None
		g_samples[batch_start:batch_end, ...] = \
			baby.step(None, batch_len, gen_only=True, z_data=batch_z, zi_data=batch_zi)
	return g_samples

'''
Evaluate encoder logits on the given dataset. **g_num**
'''
def eval_baby_en(baby, im_data, batch_size=512):
	sample_size = im_data.shape[0]
	en_logits = np.zeros([sample_size, baby.g_num])
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_im = im_data[batch_start:batch_end, ...]
		en_logits[batch_start:batch_end, ...] = \
			baby.step(batch_im, None, en_only=True)
	return en_logits

'''
Compute the accuracy of encoder network over generator ids **g_num**
'''
def eval_dataset_en(baby, im_data, im_lable, batch_size=512):
	en_logits = eval_baby_en(baby, im_data, batch_size)
	acc = np.mean((np.argmax(en_logits, axis=1) - im_lable) == 0)
	return acc

'''
1d data: mean and std of samples
'''
def eval_dataset_stat(baby, batch_size=512):
	g_mean = list()
	g_std = list()
	sample_size = 1000
	for i in range(baby.g_num):
		z = i * np.ones(sample_size)
		g_data = sample_baby_gan(baby, sample_size, z_data=z)
		g_mean.append(np.mean(g_data))
		g_std.append(np.std(g_data))
	return g_mean, g_std

'''
Training Baby GAN
'''
def train_baby_gan(baby, data_sampler):
	### dataset definition
	data_dim = baby.data_dim
	train_size = 50000

	### drawing configs
	fov = 4 ## field of view in field plot
	d_draw = 0
	g_draw = 0
	g_manifold = 0
	field_sample_size = 1000

	### training configs **vee** **g_num**
	max_itr_total = 5e5
	g_max_itr = 2e4
	d_updates = 5
	g_updates = 1
	batch_size = 32
	eval_step = eval_int

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()
	stats_logs = list()
	itrs_logs = list()
	rl_vals_logs = list()
	rl_pvals_logs = list()
	en_acc_logs = list()
	g_mean_logs = list()
	g_std_logs = list()

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	epoch = 0
	d_update_flag = True if d_updates > 0 else False
	widgets = ["baby_gan", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()
	train_dataset = data_sampler(train_size).reshape([-1, data_dim])
	#train_dataset, train_gt = \
	#	generate_normal_data(train_size, centers, stds, ratios)

	while itr_total < max_itr_total:
		### get samples from dataset 
		np.random.shuffle(train_dataset)
		#train_dataset = data_sampler(train_size)
		#	generate_circle_data(train_size)
		
		epoch += 1
		print ">>> Epoch %d started..." % epoch
		### train one epoch
		for batch_start in range(0, train_size, batch_size):
			if itr_total >= max_itr_total:
				break
			pbar.update(itr_total)
			batch_end = batch_start + batch_size
			### fetch batch data
			batch_data = train_dataset[batch_start:batch_end, :]
			fetch_batch = False
			while fetch_batch is False:
				### evaluate energy distance between real and gen distributions
				if itr_total % eval_step == 0:
					e_dist, e_norm, net_stats = eval_baby_gan(baby, data_sampler, itr_total)
					e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
					eval_logs.append([e_dist, e_dist/np.sqrt(2.0*e_norm)])
					stats_logs.append(net_stats)
					itrs_logs.append(itr_total)
					rl_vals_logs.append(list(baby.g_rl_vals))
					rl_pvals_logs.append(list(baby.g_rl_pvals))
					#z_pr = np.exp(baby.pg_temp * baby.g_rl_pvals)
					#z_pr = z_pr / np.sum(z_pr)
					#rl_pvals_logs.append(list(z_pr))
					
					### en_accuracy plots **g_num** **vee**
					acc_array = np.zeros(baby.g_num)
					sample_size = 1000
					for g in range(baby.g_num):
						z = g * np.ones(sample_size)
						z = z.astype(np.int32)
						g_samples = sample_baby_gan(baby, sample_size, z_data=z)
					#	plot_dataset_en(baby, g_samples, 'tab10', 
					#		pathname=log_path_manifold+'/data_%06d_g_%d.png' % (itr_total, g))
						acc_array[g] = eval_dataset_en(baby, g_samples, z)
					en_acc_logs.append(list(acc_array))
					
					### field plots
					'''
					g_data = sample_baby_gan(baby, field_sample_size)
					r_data = train_dataset[0:field_sample_size, ...]
					field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
					plot_field_2d(field_params, fov, 1,
						log_path_png+'/field_%06d.png' % itr_total, 
						'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total), 
						r_data=r_data, g_data=g_data)
					'''
					g_mean, g_std = eval_dataset_stat(baby)
					g_mean_logs.append(g_mean)
					g_std_logs.append(g_std)

				### discriminator update
				if d_update_flag is True:
					logs, batch_g_data = baby.step(batch_data, batch_size=None, gen_update=False)
					### logging dis results
					g_logs.append(logs[0])
					d_r_logs.append(logs[1])
					d_g_logs.append(logs[2])
					### collect gen and real data
					#g_data = sample_baby_gan(baby, field_sample_size)
					#r_data = train_dataset[0:field_sample_size, 0:data_dim]
					#r_data = r_data.reshape((r_data.shape[0], data_dim))
					### calculate and plot field of decision for dis update
					#field_params = None
					#if d_draw > 0 and d_itr % d_draw == 0:
					#	if data_dim == 1:
					#		field_params = baby_gan_field_1d(baby, -fov, fov, batch_size*10)
					#		plot_field_1d(field_params, r_data, batch_data, g_data, batch_g_data, 0,
					#			log_path_png+'/field_%06d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))    
					#	else:
					#		field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
					#		plot_field_2d(field_params, fov, r_data, batch_data, g_data, batch_g_data, 0,
					#			log_path_png+'/field_%06d.png' % itr_total, 'DIS_%d_%d_%d' % (d_itr%d_updates, g_itr, itr_total))
					d_itr += 1
					itr_total += 1
					d_update_flag = False if d_itr % d_updates == 0 else True
					fetch_batch = True
				
				### generator updates: g_updates times for each d_updates of discriminator
				elif g_updates > 0:
					logs, batch_g_data = baby.step(batch_data, batch_size=None, gen_update=True)
					### logging gen results
					g_logs.append(logs[0])
					d_r_logs.append(logs[1])
					d_g_logs.append(logs[2])
					### collect gen and real data
					#g_data = sample_baby_gan(baby, field_sample_size)
					### calculate and plot field of decision for gen update
					#if g_draw > 0 and g_itr % g_draw == 0:
					#	if data_dim == 1:
					#		if field_params is None:
					#			field_params = baby_gan_field_1d(baby, -fov, fov, batch_size*10)
					#		plot_field_1d(field_params, r_data, batch_data, g_data, batch_g_data, 0,
					#			log_path_png+'/field_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (g_itr%g_updates, g_itr, itr_total))
					#	else:
					#		if field_params is None:
					#			field_params = baby_gan_field_2d(baby, -fov, fov, -fov, fov, batch_size*10)
					#		plot_field_2d(field_params, fov, r_data, batch_data, g_data, batch_g_data, 0,
					#			log_path_png+'/field_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (g_itr%g_updates, g_itr, itr_total))
					### draw manifold of generator data
					#if g_manifold > 0 and g_itr % g_manifold == 0:
					#	plot_manifold(baby, 200, 0, log_path_manifold+'/manifold_%06d.png' % itr_total, 'GEN_%d_%d_%d' % (g_itr%g_updates, g_itr, itr_total))
					g_itr += 1
					itr_total += 1
					d_update_flag = True if g_itr % g_updates == 0 else False
				
				if itr_total >= max_itr_total:
					break

				#_, dis_confs, trace = baby.gen_consolidate(count=50)
				#print '>>> CONFS: ', dis_confs
				#print '>>> TRACE: ', trace
				#baby.reset_network('d_')

		baby.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

		### plot baby gan progress logs
		if len(eval_logs) < 2:
			continue
		g_logs_mat = np.array(g_logs)
		d_r_logs_mat = np.array(d_r_logs)
		d_g_logs_mat = np.array(d_g_logs)
		eval_logs_mat = np.array(eval_logs)
		stats_logs_mat = np.array(stats_logs)
		rl_vals_logs_mat = np.array(rl_vals_logs)
		rl_pvals_logs_mat = np.array(rl_pvals_logs)
		en_acc_logs_mat = np.array(en_acc_logs)
		g_mean_logs_mat = np.array(g_mean_logs)
		g_std_logs_mat = np.array(g_std_logs)

		g_logs_names = ['g_loss', 'g_logit_diff', 'g_out_diff', 'g_param_diff']
		d_r_logs_names = ['d_loss', 'd_param_diff', 'd_r_loss', 'r_logit_data', 'd_r_logit_diff', 'd_r_param_diff']
		d_g_logs_names = ['d_g_loss', 'g_logit_data', 'd_g_logit_diff', 'd_g_param_diff']
		eval_logs_names = ['energy_distance', 'energy_distance_norm']
		stats_logs_names = ['nan_vars', 'inf_vars', 'tiny_vars_ratio', 
							'big_vars_ratio']

		plot_time_mat(g_logs_mat, g_logs_names, 1, log_path)
		plot_time_mat(d_r_logs_mat, d_r_logs_names, 1, log_path)
		plot_time_mat(d_g_logs_mat, d_g_logs_names, 1, log_path)
		plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path, itrs=itrs_logs)
		plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path, itrs=itrs_logs)

		### plot rl_vals **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(baby.g_num):
			ax.plot(itrs_logs, rl_vals_logs_mat[:, g], label='g_%d' % g)
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('RL Q Values')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/rl_q_vals.png', dpi=300)
		plt.close(fig)
		
		### plot rl_pvals **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(baby.g_num):
			ax.plot(itrs_logs, rl_pvals_logs_mat[:, g], label='g_%d' % g)
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('RL Policy')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/rl_policy.png', dpi=300)
		plt.close(fig)

		### plot en_accs **g_num**
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(baby.g_num):
			ax.plot(itrs_logs, en_acc_logs_mat[:, g], label='g_%d' % g)
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Encoder Accuracy')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/encoder_acc.png', dpi=300)
		plt.close(fig)

		### plot g_stat_1d
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(baby.g_num):
			ax.errorbar(itrs_logs, g_mean_logs_mat[:, g], yerr=g_std_logs_mat[:,g], label='g_%d' % g,
				capsize=2, marker='.', linestyle='--')
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Generator(s)')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/gen_stat_samples.png', dpi=300)
		plt.close(fig)
		### save g_stat_1d
		with open(log_path+'/gen_stat_samples.cpk', 'wb+') as fs:
			pk.dump([itrs_logs, g_mean_logs_mat, g_std_logs_mat], fs)


def eval_baby_gan(baby, data_sampler, itr):
	### get network stats
	net_stats = baby.step(None, None, stats_only=True)
	return 0, 0, net_stats

	### dataset definition
	data_dim = baby.data_dim
	sample_size = 10000
	r_samples = data_sampler(sample_size)
		#generate_normal_data(sample_size, centers, stds, ratios)
		#generate_circle_data(sample_size)

	g_samples = sample_baby_gan(baby, sample_size)
	if data_dim > 1:
		rr_score = np.mean(np.sqrt(np.sum(np.square(r_samples[0:sample_size//2, ...] - r_samples[sample_size//2:, ...]), axis=1)))
		gg_score = np.mean(np.sqrt(np.sum(np.square(g_samples[0:sample_size//2, ...] - g_samples[sample_size//2:, ...]), axis=1)))
		rg_score = np.mean(np.sqrt(np.sum(np.square(r_samples[0:sample_size//2, ...] - g_samples[0:sample_size//2, ...]), axis=1)))
	else:
		rr_score = np.mean(np.abs(r_samples[0:sample_size//2] - r_samples[sample_size//2:]))
		gg_score = np.mean(np.abs(g_samples[0:sample_size//2] - g_samples[sample_size//2:]))
		rg_score = np.mean(np.abs(r_samples[0:sample_size//2] - g_samples[0:sample_size//2]))

	### get network stats
	net_stats = baby.step(None, None, stats_only=True)

	### draw samples **1d_datadim** **g_num** **vee**
	data_r = r_samples
	data_g = g_samples
	plot_dataset_en(baby, data_r, color_map='tab10', pathname=log_path_data+'/data_%06d_r.png' % itr)
	plot_dataset_en(baby, data_g, color_map='tab10', pathname=log_path_data+'/data_%06d_g.png' % itr)
	plot_dataset_gid(baby, sample_size, color_map='tab10', pathname=log_path_data+'/data_%06d' % itr)
	#plot_dataset([data_r, data_g], color=['r', 'b'], pathname=log_path_data+'/data_%06d.png' % itr)
	#plot_manifold_1d(baby, pathname=log_path_manifold+'/data_%06d.png' % itr)

	return 2*rg_score - rr_score - gg_score, rg_score, net_stats


if __name__ == '__main__':
	'''
	DATASET MAKING **1d_datadim**
	'''
	centers = [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
	stds = [[0.02, 0.02], [0.02, 0.02], [0.02, 0.02], [0.02, 0.02]]
	#centers = [[-0.5, 0.0], [0.5, 0.0], [0.0, 0.5], [0.0, -0.5]]
	#stds = [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]]
	#ratios = [0.2, 0.2, 0.4, 0.2]
	ratios = None
	data_dim = 1

	### function with data_size input that generates randomized training data **1d_datadim**
	data_sampler = generate_dot_data
	#data_sampler = generate_line_data
	#data_sampler = generate_circle_data
	data_r = data_sampler(50000)
	plot_dataset([data_r], color=['r'], pathname=log_path+'/real_dataset.png')

	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = tf.Session(config=config)
	
	### get a babygan instance **vee**
	baby = tf_baby_gan.TFBabyGAN(sess, data_dim)
	#baby = vee_gan.VEEGAN(sess, data_dim)
	#baby = baby_gan.BabyGAN(data_dim)
	
	### init variables
	sess.run(tf.global_variables_initializer())

	with open(log_path+'/vars_count_log.txt', 'w+') as fs:
		print >>fs, '>>> g_vars: %d --- d_vars: %d --- e_vars: %d' \
			% (baby.g_vars_count, baby.d_vars_count, baby.e_vars_count)
	'''
	GAN SETUP
	'''
	### train baby gan
	train_baby_gan(baby, data_sampler)

	### load baby
	#baby_path = '/media/evl/Public/Mahyar/dnet_logs/logs_27/run_0/snapshots/model_83333_500000.h5'
	#baby_path = '/media/evl/Public/Mahyar/ganist_logs/temp/logs_4l7ub_was_gset1_gpc1/run_0/snapshots/model_83333_500000.h5'
	#baby.load(baby_path)

	### generate sample draw
	sample_size = 10000
	data_g = sample_baby_gan(baby, sample_size)
	data_r = data_sampler(sample_size)
	plot_dataset([data_r], color=['r'], pathname=log_path+'/real_dataset.png', fov=2)
	#plot_dataset([data_g], color=['b'], pathname=log_path+'/gen_dataset.png', fov=2)
	plot_dataset_en(baby, data_g, color_map='tab10', pathname=log_path+'/gen_dataset.png', fov=2, color_bar=False)

	### eval baby gan
	#e_dist, e_norm, net_stats = eval_baby_gan(baby, centers, stds)
	#with open(log_path+'/txt_logs.txt', 'w+') as fs:
#		print >>fs, '>>> g_vars: %d --- d_vars: %d --- e_vars: %d' % (baby.g_vars_count, baby.d_vars_count, baby.e_vars_count)
#		e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
#		print >>fs, '>>> energy_distance: %f, energy_coef: %f' % (e_dist, e_dist/np.sqrt(2.0*e_norm))
#		print >>fs, '>>> nan_vars: %f, inf_vars: %f, tiny_vars: %f, big_vars: %f, count_vars: %d' % tuple(net_stats)