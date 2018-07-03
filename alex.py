import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import time


BATCH_SIZE = 100
SHUFFLE_BUFFER = 20000
TRAINING_STEPS = 50000
LEARNING_RATE_BASE = 0.003
DECAY_STEPS = 1000
LEARNING_RATE_DECAY = 0.85
MOMENTUM = 0.9
MODEL_SAVE_PATH="model/"
MODEL_NAME="alexnet_model"

HEIGHT = 32
WIDTH = 32
DEPTH = 3


def inference(features, isTraining, reuse):
	input_layer = tf.reshape(features, [-1, HEIGHT, WIDTH, DEPTH])
	with tf.variable_scope('conv1', reuse=reuse):
		conv1 = tf.layers.conv2d(
			inputs = input_layer, 
			filters = 24, 
			kernel_size = [3, 3], 
			padding = 'same',
			activation = tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	# lrn1 = tf.nn.lrn(
	# 	input = conv1,
	# 	depth_radius = 5,
	# 	bias = 2,
	# 	alpha = 0.0001,
	# 	beta = 0.75)
	pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides=2, padding='same')
	with tf.variable_scope('conv2', reuse=reuse):
		conv2 = tf.layers.conv2d(
			inputs = pool1, 
			filters = 96, 
			kernel_size = [3, 3], 
			padding = 'same',
			activation = tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	# lrn2 = tf.nn.lrn(
	# 	input = conv2,
	# 	depth_radius = 5,
	# 	bias = 2,
	# 	alpha = 0.0001,
	# 	beta = 0.75)
	pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides=1, padding='same')
	with tf.variable_scope('conv3', reuse=reuse):
		conv3 = tf.layers.conv2d(
			inputs = pool2, 
			filters = 192, 
			kernel_size = [3, 3], 
			padding = 'same',
			activation = tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	with tf.variable_scope('conv4', reuse=reuse):
		conv4 = tf.layers.conv2d(
			inputs = conv3, 
			filters = 192, 
			kernel_size = [3, 3], 
			padding = 'same',
			activation = tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	with tf.variable_scope('conv5', reuse=reuse):
		conv5 = tf.layers.conv2d(
			inputs = conv4, 
			filters = 96, 
			kernel_size = [3, 3], 
			padding = 'same',
			activation = tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	pool3 = tf.layers.max_pooling2d(inputs = conv5, pool_size = [2, 2], strides=1, padding='same')
	pool_shape = pool3.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	pool3_flat = tf.reshape(pool3, [-1, nodes])
	with tf.variable_scope('fc1', reuse=reuse):
		fc1 = tf.layers.dense(
			inputs = pool3_flat, 
			units=1024, 
			activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	dropout1 = tf.layers.dropout(inputs = fc1, rate = 0.5, training=isTraining)
	with tf.variable_scope('fc2', reuse=reuse):
		fc2 = tf.layers.dense(
			inputs = dropout1, 
			units=1024, 
			activation=tf.nn.relu,
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
	dropout2 = tf.layers.dropout(inputs = fc2, rate = 0.5, training=isTraining)
	with tf.variable_scope('fc3', reuse=reuse):
		logits = tf.layers.dense(
			inputs = dropout2, 
			units=10, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
			bias_initializer=tf.zeros_initializer(),
			reuse=reuse)
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
	

def main(argv=None):
	# train data input
	train_file = tf.train.match_filenames_once("../cifar-10-data/train.tfrecords")
	train_dataset = tf.data.TFRecordDataset(train_file)
	train_dataset = train_dataset.map(train_parser)
	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
	train_dataset = train_dataset.repeat(None)
	train_iterator = train_dataset.make_initializable_iterator()
	train_image, train_label = train_iterator.get_next()

	# test data input
	test_file = tf.train.match_filenames_once("../cifar-10-data/eval.tfrecords")
	test_dataset = tf.data.TFRecordDataset(test_file)
	test_dataset = test_dataset.map(test_parser)
	test_dataset = test_dataset.batch(1000)
	test_dataset = test_dataset.repeat(None)
	test_iterator = test_dataset.make_initializable_iterator()
	test_image, test_label = test_iterator.get_next()

	# train
	y = inference(train_image, True, False)
	global_step = tf.Variable(0, trainable=False)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=train_label)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	tf.add_to_collection('losses', cross_entropy_mean)
	loss = tf.add_n(tf.get_collection('losses'))
	correct_prediction = tf.equal(tf.argmax(y, -1, output_type=tf.int32), train_label)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		DECAY_STEPS, 
		LEARNING_RATE_DECAY,
		staircase=True)
	train_step = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(loss, global_step=global_step)

	# test
	test_logits = inference(test_image, False, True)
	test_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_logits, labels=test_label)
	test_loss = tf.reduce_mean(test_cross_entropy)
	test_correct_prediction = tf.equal(tf.argmax(test_logits, -1, output_type=tf.int32), test_label)
	test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	# saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		sess.run([train_iterator.initializer, test_iterator.initializer])
		f = open('result/result8.txt', 'a')
		time_start = time.time()
		for i in range(TRAINING_STEPS):
			_, loss_value, step, accuracy_score = sess.run([train_step, loss, global_step, accuracy])

			if i % 100 == 0:
				test_loss_value, test_accuracy_score = sess.run([test_loss, test_accuracy])
				print("step: %d, time: %g, lr: %g, loss: %g, test_loss: %g, acc: %g, test_acc: %g" % 
					(step, time.time()-time_start, learning_rate.eval(), loss_value, 
						test_loss_value, accuracy_score, test_accuracy_score))
				f.write("%d\t%g\t%g\t%g\t%g\t%g\t%g\n" % (step, time.time()-time_start, learning_rate.eval(), 
					loss_value, test_loss_value, accuracy_score, test_accuracy_score))

				# saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
		f.close()



if __name__ == '__main__':
	main()








