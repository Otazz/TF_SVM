import tensorflow as tf
import numpy as np

class Model_NonLin(object):
	def __init__(self, sess, X, y, c, gamma):
		self.sess = sess
		self.X = X
		self.y = y
		self.C = c
		self.g = gamma
		self.nFeatures = len(X[0])
		self.batch_size = len(X)

		self.x_data = tf.placeholder(shape=[None, self.nFeatures], dtype=tf.float32)

	def cross_matrices(self, tensor_a, a_inputs, tensor_b, b_inputs):
		expanded_a = tf.expand_dims(tensor_a, 1)
		expanded_b = tf.expand_dims(tensor_b, 0)
		tiled_a = tf.tile(expanded_a, tf.constant([1, b_inputs, 1]))
		tiled_b = tf.tile(expanded_b, tf.constant([a_inputs, 1, 1]))

		return [tiled_a, tiled_b]

	def gaussian_kernel(self, tensor_a, a_inputs, tensor_b, b_inputs, gamma):
		cross = self.cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs)

		kernel = tf.exp(tf.multiply(tf.reduce_sum(tf.square(
			tf.subtract(cross[0], cross[1])), reduction_indices=2),
			tf.negative(gamma)))

		return kernel


	def cost(self, training, classes, inputs, C=1, gamma=1):
		beta = tf.Variable(tf.zeros([inputs, 1]), name="beta")
		offset = tf.Variable(tf.zeros([1]), name="offset")
	
		kernel = self.gaussian_kernel(training, inputs, training, inputs, gamma)

		x = tf.reshape(tf.div(tf.matmul(tf.matmul(
			beta, kernel, transpose_a=True), beta), tf.constant([2.0])), [1])
		y = tf.subtract(tf.ones([1]), tf.multiply(classes, tf.add(
			tf.matmul(kernel, beta, transpose_a=True), offset)))
		z = tf.multiply(tf.reduce_sum(tf.reduce_max(
			tf.concat([y, tf.zeros_like(y)], 1), reduction_indices=1)),
			C)
		cost = tf.add(x, z)
		return beta, offset, cost


	def decide(self, training, training_instances, testing, testing_instances,
			   beta, offset, gamma=1):
		kernel = self.gaussian_kernel(
			testing, testing_instances, training, training_instances, gamma)

		return tf.sign(tf.add(tf.matmul(kernel, beta), offset))

	def fit(self):
		x_vals = self.X
		y_vals = self.y

		self.pos = list(y_vals).count(1)
		self.neg = list(y_vals).count(-1)

		w = [(n * self.neg/self.pos) if n == 1 else n for n in y_vals]

		
		self.Xk = tf.constant(np.array(self.X), name="X")

		y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
		self.c = tf.constant(self.C, tf.float32)
		self.gamma = tf.constant(self.g, tf.float32, name="gamma")

		self.x_d = tf.divide(tf.subtract(self.x_data, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))

		self.beta, self.offset, cost = self.cost(self.x_d, y_target, self.batch_size, 
			C=self.c, gamma=self.gamma)

		train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
		init = tf.global_variables_initializer()
		self.sess.run(init)

		for i in range(1000):
			self.sess.run(train_step, feed_dict={
			 self.x_data: x_vals, y_target: np.transpose([w])})

	def predict(self, X):
		x_r = tf.placeholder(shape=[None, self.nFeatures], dtype=tf.float32)
		x_re = tf.divide(tf.subtract(x_r, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))

		model = self.decide(self.x_d, self.batch_size, x_re, len(X), self.beta, self.offset, gamma=self.gamma)

		return self.sess.run(model, feed_dict={self.x_data: self.X,
									 x_r: X})

	def predict_nonlin(self, X, beta, offset):
		x_r = tf.placeholder(shape=[None, self.nFeatures], dtype=tf.float32)
		x_d = tf.divide(tf.subtract(self.x_data, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))
		x_re = tf.divide(tf.subtract(x_r, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))

		model = self.decide(x_d, self.batch_size, x_re, len(X), tf.transpose(np.array([beta])), offset,
			gamma=self.g)

		return self.sess.run(model, feed_dict={self.x_data: self.X,
									 x_r: X})

	def save(self):
		np.savetxt('beta.csv', self.sess.run(self.beta), delimiter=',')
		np.savetxt('offset.csv', self.sess.run(self.offset), delimiter=',')
		np.savetxt('Xk.csv', self.sess.run(self.Xk), delimiter=',')
		with open('gamma.txt', 'w') as f:
			f.write('%lf' % self.sess.run(self.gamma))


class Model_Lin(object):
	def __init__(self, sess, X, y, c):
		self.sess = sess
		self.X = X
		self.y = y
		self.c = c

	def fit(self):
		x_vals_train, y_vals_train = self.X, self.y

		self.pos = list(y_vals_train).count(1)
		self.neg = list(y_vals_train).count(-1)

		w = [(n * self.neg/self.pos) if n == 1 else n for n in y_vals_train]

		nFeatures = len(x_vals_train[0])
		batch_size = len(x_vals_train)

		self.x_data = tf.placeholder(shape=[None, nFeatures], dtype=tf.float32)
		y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

		self.Xk = tf.constant(np.array(self.X), name="X")
		self.g = tf.constant(0, name="gamma")

		x_d = tf.divide(tf.subtract(self.x_data, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))

		self.A = tf.Variable(tf.random_normal(shape=[nFeatures,1]), name='A')
		self.b = tf.Variable(tf.random_normal(shape=[1, 1]), name='b')

		model_output = tf.subtract(tf.matmul(x_d, self.A), self.b)
		l2_norm = tf.reduce_sum(self.A)

		c = tf.constant([self.c], tf.float32)

		classification_term = tf.reduce_mean(tf.maximum(0.,tf.subtract(1., tf.multiply(model_output, y_target))))
		loss = tf.add(classification_term, tf.multiply(c, l2_norm))

		prediction = tf.sign(model_output)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))

		opt = tf.train.GradientDescentOptimizer(0.001)
		train_step = opt.minimize(loss)

		init = tf.global_variables_initializer()
		self.sess.run(init)

		loss_vec = []
		train_accuracy = []
		for i in range(1000):
			rand_x = x_vals_train
			rand_y = np.transpose([w])
			self.sess.run(train_step, feed_dict={self.x_data: rand_x, y_target:rand_y})
			temp_loss = self.sess.run(loss, feed_dict={self.x_data: rand_x, y_target: rand_y})
			loss_vec.append(temp_loss)

			train_acc_temp = self.sess.run(accuracy, feed_dict={self.x_data: rand_x,
															y_target: rand_y})
			train_accuracy.append(train_acc_temp)


	def predict(self, X):
		a_r = self.sess.run(self.A)
		b_r = self.sess.run(self.b)
		x_t= tf.placeholder(shape=[None, 16], dtype=tf.float32)

		x_d = tf.divide(tf.subtract(x_t, tf.reduce_min(self.x_data)),tf.subtract(
				tf.reduce_max(self.x_data),tf.reduce_min(self.x_data)))

		pred = tf.sign(tf.matmul(tf.cast(x_d, tf.float32),a_r) + b_r)

		return self.sess.run(pred, feed_dict={self.x_data: self.X,
															x_t: X})

	def get_score(self, X, y):
		eq = 0
		for y_real, y_pred in zip(y,self.predict(X)):
			if y_real == y_pred:
				eq += 1
		return eq/len(y)

	def save(self):
		#p = params(self.sess.run(self.A), self.sess.run(self.b), self.sess.run(self.Xk), self.sess.run(self.g))
		np.savetxt('A.csv', self.sess.run(self.A), delimiter=',')
		np.savetxt('b.csv', self.sess.run(self.b), delimiter=',')
		#pickle.dump(p,open('model.p','wb'))
		#saver = tf.train.Saver()
		#saver.save(self.sess, './model')