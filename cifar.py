import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

HEIGHT = 32
WIDTH = 32
DEPTH = 3
CROP_HEIGHT = 24
CROP_WIDTH = 24


def inference(features, isTraining):
	input_layer = tf.reshape(features, [-1, CROP_HEIGHT, CROP_WIDTH, DEPTH])
	conv1 = tf.layers.conv2d(
		inputs = input_layer, 
		filters = 32, 
		kernel_size = [5, 5], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.0001),
		bias_initializer=tf.zeros_initializer())
	# lrn1 = tf.nn.lrn(
	# 	input = conv1,
	# 	depth_radius = 5,
	# 	bias = 2,
	# 	alpha = 0.0001,
	# 	beta = 0.75)
	pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides=2)
	conv2 = tf.layers.conv2d(
		inputs = pool1, 
		filters = 32, 
		kernel_size = [5, 5], 
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
	pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides=2)
	conv3 = tf.layers.conv2d(
		inputs = pool2, 
		filters = 64, 
		kernel_size = [5, 5], 
		padding = 'same',
		activation = tf.nn.relu,
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
		bias_initializer=tf.zeros_initializer())
	# conv4 = tf.layers.conv2d(
	# 	inputs = conv3, 
	# 	filters = 128, 
	# 	kernel_size = [3, 3], 
	# 	padding = 'same',
	# 	activation = tf.nn.relu,
	# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
	# 	bias_initializer=tf.zeros_initializer())
	# conv5 = tf.layers.conv2d(
	# 	inputs = conv4, 
	# 	filters = 64, 
	# 	kernel_size = [3, 3], 
	# 	padding = 'same',
	# 	activation = tf.nn.relu,
	# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
	# 	bias_initializer=tf.zeros_initializer())
	pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2, 2], strides=1, padding='same')
	pool3_flat = tf.reshape(pool3, [-1, 6*6*64])
	fc1 = tf.layers.dense(
		inputs = pool3_flat, 
		units=64, 
		activation=tf.nn.relu, 
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
		bias_initializer=tf.zeros_initializer())
	dropout1 = tf.layers.dropout(inputs = fc1, rate = 0.5, training=isTraining)
	# fc2 = tf.layers.dense(
	# 	inputs = dropout1, 
	# 	units=1024, 
	# 	activation=tf.nn.relu,
	# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
	# 	bias_initializer=tf.zeros_initializer())
	# dropout2 = tf.layers.dropout(inputs = fc2, rate = 0.5, training=isTraining)
	logits = tf.layers.dense(
		inputs = dropout1, 
		units=10, 
		kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
		bias_initializer=tf.zeros_initializer())
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
	image = tf.random_crop(image, [CROP_HEIGHT, CROP_WIDTH, DEPTH])
	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta=63)
	image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
	# 减去均值除以方差,线性缩放为零均值的单位范数:白化/标准化处理
	image = tf.image.per_image_standardization(image)
	return image, label


def test_parser(serialized_example):
	image, label = parser(serialized_example)
	image = tf.image.resize_image_with_crop_or_pad(image, CROP_HEIGHT, CROP_WIDTH)
	# 减去均值除以方差,线性缩放为零均值的单位范数:白化/标准化处理
	image = tf.image.per_image_standardization(image)
	return image, label
	


