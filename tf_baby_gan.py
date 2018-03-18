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

class TFBabyGAN:
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
		self.pg_lr = 1e-2
		self.pg_beta1 = 0.5
		self.pg_beta2 = 0.5

		### network parameters
		self.batch_size = 128
		self.z_dim = 100
		self.man_dim = 0
		self.g_num = 4
		self.z_range = 1.0
		self.data_dim = data_dim
		self.mm_loss_weight = 0.0
		self.gp_loss_weight = 10.0
		self.en_loss_weight = 1.0
		self.rl_lr = 0.99
		self.pg_q_lr = 0.99
		self.pg_temp = 1.0
		self.g_rl_vals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.g_rl_pvals = 0. * np.ones(self.g_num, dtype=np_dtype)
		self.d_loss_type = 'log'
		self.g_loss_type = 'mod'
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
		self.g_layer = self.build_gen(self.z_input, self.zi_input, self.g_act, self.train_phase)

		### build discriminator
		self.r_logits, self.r_hidden = self.build_dis(self.im_input, self.d_act, self.train_phase)
		self.g_logits, self.g_hidden = self.build_dis(self.g_layer, self.d_act, self.train_phase, reuse=True)
		self.r_en_logits = self.build_encoder(self.r_hidden, self.d_act, self.train_phase)
		self.g_en_logits = self.build_encoder(self.g_hidden, self.d_act, self.train_phase, reuse=True)

		### real gen manifold interpolation
		rg_layer = (1.0 - self.e_input) * self.g_layer + self.e_input * self.im_input
		self.rg_logits, _ = self.build_dis(rg_layer, self.d_act, self.train_phase, reuse=True)

		### build d losses
		if self.d_loss_type == 'log':
			self.d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype))
			self.d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
			self.d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.rg_logits, labels=tf.ones_like(self.rg_logits, tf_dtype))
		elif self.d_loss_type == 'was':
			self.d_r_loss = -self.r_logits 
			self.d_g_loss = self.g_logits
			self.d_rg_loss = -self.rg_logits
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### gradient penalty
		### NaN free norm gradient
		rg_grad = tf.gradients(self.rg_logits, rg_layer)
		rg_grad_flat = tf.reshape(rg_grad, [-1, np.prod(self.data_dim)])
		rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 0.
		rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
		rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
		rg_grad_norm = tf.where(rg_grad_ok, 
			tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
		gp_loss = tf.square(rg_grad_norm - 1.0)

		### generated encoder loss given z_input has generator ids **g_num**
		self.g_en_loss = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			logits=self.g_en_logits)

		### d loss combination **g_num**
		self.d_loss_mean = tf.reduce_mean(self.d_r_loss + self.d_g_loss)
		self.d_loss_total = self.d_loss_mean + self.gp_loss_weight * tf.reduce_mean(gp_loss)

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
		
		### generated encoder loss: lower bound on mutual_info(z_input, generator id) **g_num**
		self.g_en_loss = tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			logits=self.g_en_logits)

		### real encoder entropy: entropy of g_id given real image, marginal entropy of g_id **g_num**
		self.r_en_h = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(self.r_en_logits) * tf.nn.log_softmax(self.r_en_logits), axis=1))
		r_en_marg_pr = tf.reduce_mean(tf.nn.softmax(self.r_en_logits), axis=0)
		self.r_en_marg_hlb = -tf.reduce_sum(r_en_marg_pr * tf.log(r_en_marg_pr + 1e-8))

		### mean matching
		mm_loss = tf.reduce_mean(tf.square(tf.reduce_mean(self.g_layer, axis=0) - \
			tf.reduce_mean(self.im_input, axis=0)), axis=None)

		### discounter
		self.rl_counter = tf.get_variable('rl_counter', dtype=tf_dtype,
			initializer=1.0)

		### g loss combination **g_num**
		#self.g_loss_total = self.g_loss_mean + tf.where(self.rl_counter < 0.1, 
		#	self.rl_counter * self.en_loss_weight * tf.reduce_mean(self.g_en_loss), 
		#	self.en_loss_weight * tf.reduce_mean(self.g_en_loss))
		self.g_loss_total = self.g_loss_mean + self.en_loss_weight * tf.reduce_mean(self.g_en_loss)

		### e loss combination
		self.en_loss_total = tf.reduce_mean(self.g_en_loss) + \
			0. * self.r_en_h + 0.* -self.r_en_marg_hlb

		### collect params
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
		self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")

		'''	
		###consolidatoin loss
		con_loss = 0.0
		con_mem_ops = list()
		con_info_ops = list()
		self.con_mems = list()
		self.con_infos = list()
		with tf.variable_scope('consolid'):
			self.trace_counter = tf.get_variable('trace_counter', [1], dtype=tf.int32, initializer=tf.constant_initializer(0))
			trace_id = self.trace_counter[0] % self.con_trace_size
			for v in self.g_vars:
				shape = v.get_shape().as_list()
				v_mem = tf.get_variable(v.name[:-2], [self.con_trace_size] + shape, initializer=tf.constant_initializer(0.0))
				v_info = tf.get_variable(v.name[:-2]+'_info', [self.con_trace_size] + shape, initializer=tf.constant_initializer(0.0))
				con_loss += tf.reduce_sum(v_info * tf.square(v - v_mem), axis=None)
				self.con_mems.append(v_mem)
				self.con_infos.append(v_info)

				### batch separated gradients wrt weights: E_z[(grad_w D(x))^2]
				flat_logits = tf.reshape(self.g_logits, [-1])
				fl_size = tf.shape(flat_logits)[0]
				fl_pad = tf.pad(flat_logits, [[0, self.batch_size - fl_size]], 'CONSTANT')
				fl_prsq = tf.square(tf.sigmoid(fl_pad))
				fl_diag = tf.diag(fl_pad)
				fl_grads = 0.0
				for i in range(self.batch_size):
					fl_grads += tf.square(tf.gradients(fl_diag[i, ...], v)[0])
					#fl_grads += fl_prsq[i] * tf.square(tf.gradients(fl_diag[i, ...], v)[0])
				fl_grads = fl_grads / tf.cast(fl_size, tf_dtype)

				### ops tp update consolidation variables
				info_decay = v_info.assign(v_info * self.con_decay)
				with tf.control_dependencies([info_decay]):
					con_mem_ops.append(v_mem[trace_id, ...].assign(v))
					con_info_ops.append(v_info[trace_id, ...].assign(fl_grads))

			with tf.control_dependencies(con_mem_ops + con_info_ops):
				self.con_trace_update = self.trace_counter.assign_add([1])

		self.g_loss = self.g_loss + self.mm_loss_weight * mm_loss + self.con_loss_weight * con_loss
		'''

		### logs
		#r_logits_mean = tf.reduce_mean(self.r_logits, axis=None)
		#g_logits_mean = tf.reduce_mean(self.g_logits, axis=None)
		#d_r_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_r_loss, self.r_logits)), axis=None))
		#d_g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_g_loss, self.g_logits)), axis=None))
		#g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_logits)), axis=None))
		#g_out_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_layer)), axis=None))
		
		'''
		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(
				tf.square(tf.gradients(tf.reduce_mean(self.d_r_loss), v)), axis=None))
		d_r_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(
				tf.square(tf.gradients(tf.reduce_mean(self.d_g_loss), v)), axis=None))
		d_g_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.d_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(
				tf.square(tf.gradients(self.d_loss_mean, v)), axis=None))
		d_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf_dtype)
		for v in self.g_vars:
			if 'bn_' in v.name:
				continue
			diff = diff + tf.sqrt(tf.reduce_mean(
				tf.square(tf.gradients(self.g_loss_mean, v)), axis=None))
		g_param_diff = 1.0 * diff / len(self.g_vars)
		'''

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
		self.nan_vars /= self.count_vars 
		self.inf_vars /= self.count_vars
		self.zero_vars /= self.count_vars
		self.big_vars /= self.count_vars

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
				self.en_loss_total, var_list=self.e_vars)

		### summaries
		self.d_r_logs = [self.d_loss_mean]#, d_param_diff, self.d_r_loss, r_logits_mean, d_r_logits_diff, d_r_param_diff]
		self.d_g_logs = [tf.reduce_mean(self.d_g_loss)]#, g_logits_mean, d_g_logits_diff, d_g_param_diff]
		self.g_logs = [self.g_loss_mean]#, g_logits_diff, g_out_diff, g_param_diff]	

		### Policy gradient updates **g_num**
		self.pg_var = tf.get_variable('pg_var', dtype=tf_dtype,
			initializer=self.g_rl_vals)
		self.pg_q = tf.get_variable('pg_q', dtype=tf_dtype,
			initializer=self.g_rl_vals)
		self.pg_base = tf.get_variable('pg_base', dtype=tf_dtype,
			initializer=0.0)
		self.pg_var_flat = self.pg_temp * tf.reshape(self.pg_var, [1, -1])
		
		### log p(x) for the selected policy at each batch location
		log_soft_policy = -tf.nn.softmax_cross_entropy_with_logits(
			labels=tf.one_hot(tf.reshape(self.z_input, [-1]), self.g_num, dtype=tf_dtype), 
			logits=tf.tile(self.pg_var_flat, tf.shape(tf.reshape(self.z_input, [-1, 1]))))
		
		self.gi_h = -tf.reduce_sum(tf.nn.softmax(self.pg_var) * tf.nn.log_softmax(self.pg_var))
		
		### policy gradient reward
		#pg_reward = tf.reshape(self.d_g_loss, [-1]) - 0. * tf.reshape(self.g_en_loss, [-1])
		pg_reward = tf.reduce_mean(self.r_en_logits, axis=0)
		print '>>> pg_reward shape:', pg_reward.get_shape()
		
		### critic update (q values update)
		#pg_q_z = tf.gather(self.pg_q, tf.reshape(self.z_input, [-1]))
		#pg_q_opt = tf.scatter_update(self.pg_q, tf.reshape(self.z_input, [-1]), 
		#		self.pg_q_lr*pg_q_z + (1-self.pg_q_lr) * pg_reward)
		rl_counter_opt = tf.assign(self.rl_counter, self.rl_counter * 0.999)
		
		### r_en_logits as q values
		pg_q_opt = tf.assign(self.pg_q, self.pg_q_lr*self.pg_q + \
			(1-self.pg_q_lr) * pg_reward)

		### actor update (p values update)
		with tf.control_dependencies([pg_q_opt, rl_counter_opt]):
			pg_q_zu = tf.gather(self.pg_q, tf.reshape(self.z_input, [-1]))
			pg_loss_total = -tf.reduce_mean(log_soft_policy * pg_q_zu) + \
				1000. * self.rl_counter * -self.gi_h

		#self.pg_opt = tf.train.AdamOptimizer(
		#		self.pg_lr, beta1=self.pg_beta1, beta2=self.pg_beta2).minimize(
		#		pg_loss_total, var_list=[self.pg_var])
		self.pg_opt = tf.train.GradientDescentOptimizer(self.pg_lr).minimize(
			pg_loss_total, var_list=[self.pg_var])

	def build_gen(self, z, zi, act, train_phase):
		ol = list()
		with tf.variable_scope('g_net'):
			for gi in range(self.g_num):
				with tf.variable_scope('gnum_%d' % gi):
					### **g_num**
					zi = tf.random_uniform([tf.shape(z)[0], self.z_dim], 
						minval=-self.z_range, maxval=self.z_range, dtype=tf_dtype)
					bn = tf.contrib.layers.batch_norm
			
					### fully connected from hidden z 44128 to image shape
					h1 = act(bn(dense(zi, 128//4, scope='fc1')))
					h2 = act(bn(dense(h1, 64//4, scope='fc2')))
					h3 = dense(h2, self.data_dim, scope='fco')
					
					### output activation to bring data values in (-1,1)
					ol.append(h3)

			z_1_hot = tf.reshape(tf.one_hot(z, self.g_num, dtype=tf_dtype), [-1, self.g_num, 1])
			z_map = tf.tile(z_1_hot, [1, 1, self.data_dim])
			os = tf.stack(ol, axis=1)
			o = tf.reduce_sum(os * z_map, axis=1)
			return o

	def build_dis(self, data_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('d_net'):
			h1 = act(dense(data_layer, 64, scope='fc1', reuse=reuse))
			h2 = act(dense(h1, 128, scope='fc2', reuse=reuse))
			o = dense(h2, 1, scope='fco', reuse=reuse)
			return o, h2

	def build_encoder(self, hidden_layer, act, train_phase, reuse=False):
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('e_net'):
			with tf.variable_scope('encoder'):
				flat = hidden_layer
				flat = act(bn(dense(flat, 128, scope='fc', reuse=reuse), 
					reuse=reuse, scope='bf1', is_training=train_phase))
				o = dense(flat, self.g_num, scope='fco', reuse=reuse)
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
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars, self.count_vars]
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
			en_logits = self.sess.run(self.r_en_logits, feed_dict=feed_dict)
			return en_logits

		### sample z from uniform (-1,1)
		if zi_data is None:
			zi_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
				size=[batch_size, self.z_dim])
			#z_data = np.random.uniform(low=-self.z_range, high=self.z_range, 
			#	size=[batch_size, self.z_dim-self.man_dim])
			#z_data = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, self.z_dim))
			#if self.man_dim > 0:
				### select manifold of each random point (1 hot)
			#	man_id = np.random.choice(self.man_dim, batch_size)
			#	z_man = np.zeros((batch_size, self.man_dim))
			#	z_man[range(batch_size), man_id] = 1.
			#	z_data = np.concatenate([z_data, z_man], axis=1)
		zi_data = zi_data.astype(np_dtype)

		### multiple generator uses z_data to select gen **g_num**
		if z_data is None:
			#g_th = min(1 + self.rl_counter // 1000, self.g_num)
			#g_th = self.g_num
			#z_pr = np.exp(self.pg_temp * self.g_rl_pvals[:g_th])
			#z_pr = z_pr / np.sum(z_pr)
			#z_data = np.random.choice(g_th, size=batch_size, p=z_pr)
			z_data = np.random.randint(low=0, high=self.g_num, size=batch_size)

		#z_data = z_data.astype(np_dtype)

		### only forward generator on z
		if gen_only:
			feed_dict = {self.z_input: z_data, self.zi_input: zi_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log
		feed_dict = {self.im_input:batch_data, self.z_input: z_data, self.zi_input: zi_data,
			self.e_input: e_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, 
				self.d_g_logs, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			#z_data_delta = np.random.uniform(low=-0.1, high=0.1, 
			#	size=[batch_size, self.z_dim-self.man_dim])
			#z_data_delta = np.pad(z_data_delta, ((0, 0), (0, self.man_dim)), 'constant')
			#z_data = z_data[0, ...] + z_data_delta
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, 
				self.d_g_logs, self.g_opt, self.e_opt, self.pg_opt]
				#self.r_en_h, self.r_en_marg_hlb, self.gi_h, self.g_en_loss, self.rl_counter]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			self.g_rl_vals, self.g_rl_pvals = self.sess.run((self.pg_q, self.pg_var), feed_dict={})
			#print '>>> rl_counter: ', res_list[-1]
			### RL value updates
			#self.g_rl_vals[z_data] += (1-self.rl_lr) * \
			#	(-res_list[3][:,0] - self.g_rl_vals[z_data])
			#self.g_rl_vals += 1e-3
			#self.rl_counter += 1

			#self.sess.run(self.con_trace_update, feed_dict=feed_dict)

			#con_res = self.sess.run([self.trace_counter, self.con_mems, self.con_infos])
			#with open('/home/mahyar/con_results/con_%d.cpk' % con_res[0][0], 'wb+') as fs:
			#	pk.dump(con_res, fs)

		return tuple(res_list[1:]), res_list[0]
