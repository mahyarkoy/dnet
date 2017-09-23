import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float64,
								 tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [output_size], tf.float64,
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

class TFBabyGAN:
	def __init__(self, data_dim):

		# s_data must have zero pads at the end to represent gen class, columns are features
		self.g_lr = 1e-4
		self.g_beta = 0.5
		self.d_lr = 1e-4
		self.d_beta = 0.5

		### running
		self.gpu_id = 0

		### network parameters
		self.z_dim = 256
		self.z_range = 1.0
		self.data_dim = data_dim
		self.d_loss_type = 'log'
		self.g_loss_type = 'mod'
		self.d_act = tf.nn.tanh
		self.g_act = tf.nn.tanh
		#self.d_act = lrelu
		#self.g_act = tf.nn.relu

		### init graph and session
		self.build_graph()
		self.start_session()

	def __del__(self):
		self.end_session()

	def build_graph(self):
		### define placeholders for image and label inputs
		self.im_input = tf.placeholder(tf.float64, [None, self.data_dim], name='im_input')
		self.z_input = tf.placeholder(tf.float64, [None, self.z_dim], name='z_input')

		### build generator
		self.g_layer = self.build_gen(self.z_input, self.g_act)

		### build discriminator
		self.r_logits = self.build_dis(self.im_input, self.d_act)
		self.g_logits = self.build_dis(self.g_layer, self.d_act, reuse=True)

		### build d losses
		if self.d_loss_type == 'log':
			self.d_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf.float64)))
			self.d_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf.float64)))
		elif self.d_loss_type == 'was':
			self.d_r_loss = -tf.reduce_mean(self.r_logits)
			self.d_g_loss = tf.reduce_mean(self.g_logits)
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)
		self.d_loss = self.d_r_loss + self.d_g_loss

		### build g loss
		if self.g_loss_type == 'log':
			self.g_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf.float64)))
		elif self.g_loss_type == 'mod':
			self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf.float64)))
		elif self.g_loss_type == 'was':
			self.g_loss = -tf.reduce_mean(self.g_logits)
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

		### collect params
		self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
		self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")

		### logs
		r_logits_mean = tf.reduce_mean(self.r_logits)
		g_logits_mean = tf.reduce_mean(self.g_logits)
		d_r_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_r_loss, self.r_logits))))
		d_g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_g_loss, self.g_logits))))
		g_logits_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_logits))))
		g_out_diff = tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, self.g_layer))))
		
		diff = tf.zeros((1,), tf.float64)
		for v in self.d_vars:
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_r_loss, v))))
		d_r_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf.float64)
		for v in self.d_vars:
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.d_g_loss, v))))
		d_g_param_diff = 1.0 * diff / len(self.d_vars)

		diff = tf.zeros((1,), tf.float64)
		for v in self.g_vars:
			diff = diff + tf.sqrt(tf.reduce_mean(tf.square(tf.gradients(self.g_loss, v))))
		g_param_diff = 1.0 * diff / len(self.g_vars)

		### build optimizers
		self.g_opt = tf.train.AdamOptimizer(self.g_lr, beta1=self.g_beta, beta2=self.g_beta).minimize(self.g_loss, var_list=self.g_vars)
		self.d_opt = tf.train.AdamOptimizer(self.d_lr, beta1=self.d_beta, beta2=self.d_beta).minimize(self.d_loss, var_list=self.d_vars)

		### summaries
		self.d_r_logs = [self.d_r_loss, r_logits_mean, d_r_logits_diff, d_r_param_diff]
		self.d_g_logs = [self.d_g_loss, g_logits_mean, d_g_logits_diff, d_g_param_diff]
		self.g_logs = [self.g_loss, g_logits_diff, g_out_diff, g_param_diff]
		

	def build_gen(self, z, act):
		h1_size = 1024
		h2_size = 1024
		with tf.variable_scope('g_net'):
			h1 = linear(z, h1_size, scope='fc1')
			h1 = act(h1)

			h2 = linear(h1, h2_size, scope='fc2')
			h2 = act(h2)

			o = linear(h2, self.data_dim, scope='fco')
			return o

	def build_dis(self, data_layer, act, reuse=False):
		h1_size = 1024
		h2_size = 1024
		with tf.variable_scope('d_net', reuse=reuse):
			h1 = linear(data_layer, h1_size, scope='fc1')
			h1 = act(h1)

			h2 = linear(h1, h2_size, scope='fc2')
			h2 = act(h2)

			o = linear(h2, 1, scope='fco')
			return o

	def start_session(self):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
		config = tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options)
		self.saver = tf.train.Saver(tf.global_variables(), keep_checkpoint_every_n_hours=1)
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

	def end_session(self):
		self.sess.close()

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def step(self, batch_data, batch_size, gen_update=False, dis_only=False, gen_only=False):
		### only forward discriminator on batch_data
		if dis_only:
			feed_dict = {self.im_input: batch_data}
			u_logits = self.sess.run(self.r_logits, feed_dict=feed_dict)
			return u_logits.flatten()

		### sample z from uniform (-1,1)
		z_data = np.random.uniform(low=-self.z_range, high=self.z_range, size=(batch_size, self.z_dim))

		### only forward generator on z
		if gen_only:
			feed_dict = {self.z_input: z_data}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log
		feed_dict = {self.im_input:batch_data, self.z_input: z_data}
		if not gen_update:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, self.d_g_logs, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.g_logs, self.d_r_logs, self.d_g_logs, self.g_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)

		return tuple(res_list[1:]), res_list[0]