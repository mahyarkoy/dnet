import numpy as np
import tensorflow as tf
import os
import cPickle as pk

tf_dtype = tf.float32
np_dtype = 'float32'

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf_dtype,
								 tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [output_size], tf_dtype,
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

def dense_batch(x, h_size, scope, phase, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
    with tf.variable_scope(scope):
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn_'+str(reuse))
    return h2

def dense(x, h_size, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
        #h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
        #h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    return h1

class VEEGAN:
	def __init__(self, sess, data_dim):
		self.sess = sess

		### optimization parameters
		self.g_lr = 2e-4
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 2e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5
		self.e_lr = 2e-4
		self.e_beta1 = 0.9
		self.e_beta2 = 0.999
		self.pg_lr = 1e-3
		self.pg_beta1 = 0.5
		self.pg_beta2 = 0.5

		### network parameters
		self.batch_size = 128
		self.z_dim = 100
		self.man_dim = 0
		self.g_num = 1
		self.z_range = 1.0
		self.data_dim = data_dim
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 10.0
		self.en_loss_weight = 1.0
		self.rl_lr = 0.99
		self.pg_q_lr = 0.01
		self.pg_temp = 1.0
		self.g_rl_vals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.g_rl_pvals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.d_loss_type = 'log'
		self.g_loss_type = 'log'
		#self.d_act = tf.tanh
		#self.g_act = tf.tanh
		self.d_act = lrelu
		self.g_act = tf.nn.relu

		### consolidation params
		self.con_loss_weight = 0.
		self.con_trace_size = 10
		self.con_decay = 0.9

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		### define placeholders for image and label inputs
		self.im_input = tf.placeholder(tf_dtype, [None, self.data_dim], name='im_input')
		#self.z_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='z_input')
		#self.z_input = tf.placeholder(tf_dtype, [None, self.g_num, self.data_dim], name='z_input')
		self.z_input = tf.placeholder(tf.int32, [None], name='z_input')
		self.zi_input = tf.placeholder(tf_dtype, [None, self.z_dim], name='zi_input')
		self.e_input = tf.placeholder(tf_dtype, [None, 1], name='e_input')
		self.train_phase = tf.placeholder(tf.bool, name='phase')

		### build generator
		self.g_layer = self.build_gen(self.zi_input, self.g_act, self.train_phase)

		### build encoder (reconstructor)
		self.r_en_layer = self.build_encoder(self.im_input, self.d_act, self.train_phase)
		self.g_en_layer = self.build_encoder(self.g_layer, self.d_act, self.train_phase, reuse=True)

		### build joint discriminator
		self.r_logits = self.build_dis(self.im_input, self.r_en_layer, self.d_act, self.train_phase)
		self.g_logits = self.build_dis(self.g_layer, self.zi_input, self.d_act, self.train_phase, reuse=True)

		### build d losses
		if self.d_loss_type == 'log':
			self.d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype))
			self.d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
		elif self.d_loss_type == 'was':
			self.d_r_loss = -self.r_logits 
			self.d_g_loss = self.g_logits
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### d loss combination **g_num**
		self.d_loss_mean = tf.reduce_mean(self.d_r_loss + self.d_g_loss)
		self.d_loss_total = self.d_loss_mean

		### build g loss
		if self.g_loss_type == 'log':
			self.g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
		elif self.g_loss_type == 'mod':
			self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf_dtype))
		elif self.g_loss_type == 'was':
			self.g_loss = -self.g_logits
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

		self.g_loss_mean = tf.reduce_mean(self.g_loss, axis=None)

		### encode/reconstruction loss
		self.e_loss_mean = tf.reduce_mean(tf.reduce_sum(tf.square(self.zi_input - self.g_en_layer), axis=1))
		
		### g loss combination **g_num**
		self.g_loss_total = self.g_loss_mean + self.en_loss_weight * self.e_loss_mean

		### collect params
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
		self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")

		### compute stat of weights
		self.nan_vars = 0.
		self.inf_vars = 0.
		self.zero_vars = 0.
		self.big_vars = 0.
		self.count_vars = 0
		for v in self.g_vars + self.d_vars:
			self.nan_vars += tf.reduce_sum(tf.cast(tf.is_nan(v), tf_dtype))
			self.inf_vars += tf.reduce_sum(tf.cast(tf.is_inf(v), tf_dtype))
			self.zero_vars += tf.reduce_sum(tf.cast(tf.square(v) < 1e-6, tf_dtype))
			self.big_vars += tf.reduce_sum(tf.cast(tf.square(v) > 1.0, tf_dtype))
			self.count_vars += tf.reduce_prod(v.get_shape())
		self.count_vars = tf.cast(self.count_vars, tf_dtype)
		#self.nan_vars /= self.count_vars
		#self.inf_vars /= self.count_vars
		self.zero_vars /= self.count_vars
		self.big_vars /= self.count_vars

		self.g_vars_count = 0
		self.d_vars_count = 0
		self.e_vars_count = 0
		for v in self.g_vars:
			self.g_vars_count += int(np.prod(v.get_shape()))
		for v in self.d_vars:
			self.d_vars_count += int(np.prod(v.get_shape()))
		for v in self.e_vars:
			self.e_vars_count += int(np.prod(v.get_shape()))

		### build optimizers
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		print '>>> update_ops list: ', update_ops
		with tf.control_dependencies(update_ops):
			self.g_opt = tf.train.AdamOptimizer(
				self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
				self.g_loss_total, var_list=self.g_vars)
			self.d_opt = tf.train.AdamOptimizer(
				self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
				self.d_loss_total, var_list=self.d_vars)
			self.e_opt = tf.train.AdamOptimizer(
				self.e_lr, beta1=self.e_beta1, beta2=self.e_beta2).minimize(
				self.e_loss_mean, var_list=self.e_vars)

		### summaries
		self.d_r_logs = [self.d_loss_mean]#, d_param_diff, self.d_r_loss, r_logits_mean, d_r_logits_diff, d_r_param_diff]
		self.d_g_logs = [tf.reduce_mean(self.d_g_loss)]#, g_logits_mean, d_g_logits_diff, d_g_param_diff]
		self.g_logs = [self.g_loss_mean]#, g_logits_diff, g_out_diff, g_param_diff]	

	def build_gen(self, zi, act, train_phase):
		with tf.variable_scope('g_net'):
			bn = tf.contrib.layers.batch_norm
			### fully connected from hidden z 44128 to image shape
			h1 = act(bn(dense(zi, 128, scope='fc1')))
			h2 = act(bn(dense(h1, 64, scope='fc2')))
			o = dense(h2, self.data_dim, scope='fco')
			return o

	def build_dis(self, data_layer, z_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('d_net'):
			h1 = act(dense(data_layer, 64, scope='fc1', reuse=reuse))
			h2 = act(bn(dense(h1, 128, scope='fc2', reuse=reuse), 
				reuse=reuse, scope='bf2'))
			h2_flat = tf.contrib.layers.flatten(h2)
			concat = tf.concat([h2_flat, z_layer], axis=1)
			h3 = act(dense(concat, 128, scope='fc3', reuse=reuse))
			o = dense(h3, 1, scope='fco', reuse=reuse)
			return o

	def build_encoder(self, data_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('e_net'):
			h1 = act(bn(dense(data_layer, 64, scope='fc1', reuse=reuse), 
				reuse=reuse, scope='bf1'))
			h2 = act(bn(dense(h1, 128, scope='fc2', reuse=reuse), 
				reuse=reuse, scope='bf2'))
			o = dense(h2, 100, scope='fco', reuse=reuse)
			return o

	def start_session(self):
		self.saver = tf.train.Saver(tf.global_variables(), 
			keep_checkpoint_every_n_hours=1, max_to_keep=5)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def step(self, batch_data, batch_size, gen_update=False, 
		dis_only=False, gen_only=False, stats_only=False, en_only=False, z_data=None, zi_data=None):
		batch_size = batch_data.shape[0] if batch_data is not None else batch_size
		batch_data = batch_data.astype(np_dtype) if batch_data is not None else None
		
		### inf, nans, tiny and big vars stats
		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		### sample e from uniform (0,1): for gp penalty in WGAN
		e_data = np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
		e_data = e_data.astype(np_dtype)

		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### only forward discriminator on batch_data
		if en_only:
			feed_dict = {self.im_input: batch_data, self.train_phase: False}
			en_logits = self.sess.run(self.r_en_layer, feed_dict=feed_dict)
			return en_logits

		### sample z from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
		zi_data = zi_data.astype(np_dtype)

		### only forward generator on z
		if gen_only:
			feed_dict = {self.zi_input: zi_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log
		feed_dict = {self.im_input:batch_data, self.zi_input: zi_data,
			self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, 
				self.d_g_logs, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, 
				self.d_g_logs, self.g_opt, self.e_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)

		return tuple(res_list[1:]), res_list[0]
