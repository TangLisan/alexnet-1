import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

HEIGHT = 32
WIDTH = 32
DEPTH = 3


def inference(features, isTraining):
	input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, DEPTH])
	conv1 = tf.layers.conv2d(
		inputs = input_layer, 
		filters = 24, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	# lrn1 = tf.nn.lrn(
	# 	input = conv1,
	# 	depth_radius = 5,
	# 	bias = 2,
	# 	alpha = 0.0001,
	# 	beta = 0.75)
	pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides=1, padding='same')
	conv2 = tf.layers.conv2d(
		inputs = pool1, 
		filters = 64, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	# lrn2 = tf.nn.lrn(
	# 	input = conv2,
	# 	depth_radius = 5,
	# 	bias = 2,
	# 	alpha = 0.0001,
	# 	beta = 0.75)
	pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides=1, padding='same')
	conv3 = tf.layers.conv2d(
		inputs = pool2, 
		filters = 96, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	conv4 = tf.layers.conv2d(
		inputs = conv3, 
		filters = 96, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	conv5 = tf.layers.conv2d(
		inputs = conv4, 
		filters = 64, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	pool3 = tf.layers.max_pooling2d(inputs = conv5, pool_size = [2, 2], strides=1, padding='same')
	pool3_flat = tf.reshape(pool3, [-1, 32*32*64])
	fc1 = tf.layers.dense(
		inputs = pool3_flat, 
		units=1024, 
		activation=tf.nn.relu, 
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	dropout1 = tf.layers.dropout(inputs = fc1, rate = 0.5, training=isTraining)
	fc2 = tf.layers.dense(
		inputs = dropout1, 
		units=1024, 
		activation=tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	dropout2 = tf.layers.dropout(inputs = fc2, rate = 0.5, training=isTraining)
	logits = tf.layers.dense(
		inputs = dropout2, 
		units=10, 
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	return logits


def inference2(features, isTraining, reuse):
	input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, DEPTH])

	with tf.variable_scope('layer1-conv1', reuse=reuse):
		conv1_weights = tf.get_variable(
			"weight", [3, 3, 3, 24],
			initializer=tf.truncated_normal_initializer(stddev=0.01))
		conv1_biases = tf.get_variable("bias", [24], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_layer, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

	with tf.name_scope("layer2-pool1"):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,1,1,1],padding="SAME")

	with tf.variable_scope('layer3-conv2', reuse=reuse):
		conv2_weights = tf.get_variable(
			"weight", [3, 3, 24, 64],
			initializer=tf.truncated_normal_initializer(stddev=0.01))
		conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	with tf.name_scope("layer4-pool2"):
		pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,1,1,1],padding="SAME")

	with tf.variable_scope('layer5-conv3', reuse=reuse):
		conv3_weights = tf.get_variable(
			"weight", [3, 3, 64, 96],
			initializer=tf.truncated_normal_initializer(stddev=0.01))
		conv3_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

	with tf.variable_scope('layer6-conv4', reuse=reuse):
		conv4_weights = tf.get_variable(
			"weight", [3, 3, 96, 96],
			initializer=tf.truncated_normal_initializer(stddev=0.01))
		conv4_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(relu3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

	with tf.variable_scope('layer7-conv5', reuse=reuse):
		conv5_weights = tf.get_variable(
			"weight", [3, 3, 96, 64],
			initializer=tf.truncated_normal_initializer(stddev=0.01))
		conv5_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
		conv5 = tf.nn.conv2d(relu4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

	with tf.name_scope("layer8-pool3"):
		pool3 = tf.nn.max_pool(relu5, ksize = [1,2,2,1],strides=[1,1,1,1],padding="SAME")
		pool_shape = pool3.get_shape().as_list()
		nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
		reshaped = tf.reshape(pool3, [-1, nodes])

	with tf.variable_scope('layer9-fc1', reuse=reuse):
		fc1_weights = tf.get_variable("weight", [nodes, 1024],
									  initializer=tf.truncated_normal_initializer(stddev=0.01))
		fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.0))
		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		if isTraining: fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.variable_scope('layer10-fc2', reuse=reuse):
		fc2_weights = tf.get_variable("weight", [1024, 1024],
									  initializer=tf.truncated_normal_initializer(stddev=0.01))
		fc2_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.0))
		fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
		if isTraining: fc2 = tf.nn.dropout(fc2, 0.5)

	with tf.variable_scope('layer11-fc3', reuse=reuse):
		fc3_weights = tf.get_variable("weight", [1024, 10],
									  initializer=tf.truncated_normal_initializer(stddev=0.01))
		fc3_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.0))
		logits = tf.nn.relu(tf.matmul(fc2, fc3_weights) + fc3_biases)
	return logits


def parser(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64)
		})
	image = tf.decode_raw(features['image'], tf.uint8)
	image.set_shape([DEPTH * HEIGHT * WIDTH])
	# Reshape from [depth * height * width] to [depth, height, width].
	image = tf.cast(tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),tf.float32)
	label = tf.cast(features['label'], tf.int32)
	return image, label


def train_parser(serialized_example):
	image, label = parser(serialized_example)
	# Pad 4 pixels on each dimension of feature map, done in mini-batch
	image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
	image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta=63)
	image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
	# 减去均值除以方差,线性缩放为零均值的单位范数:白化/标准化处理
	# image = tf.image.per_image_standardization(image)
	return image, label


def test_parser(serialized_example):
	image, label = parser(serialized_example)
	# 减去均值除以方差,线性缩放为零均值的单位范数:白化/标准化处理
	# image = tf.image.per_image_standardization(image)
	return image, label
	


