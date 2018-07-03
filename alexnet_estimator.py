import numpy as np
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

HEIGHT = 32
WIDTH = 32
DEPTH = 3
BATCH_SIZE = 100
SHUFFLE_BUFFER = 20000
NUM_EPOCHS = 10


tf.logging.set_verbosity(tf.logging.INFO)


def alex_net(features, labels, mode):
	input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
	conv1 = tf.layers.conv2d(
		inputs = input_layer, 
		filters = 32, 
		kernel_size = [5, 5], 
		padding = 'same',
		activation = tf.nn.relu)
	lrn1 = tf.nn.lrn(
		input = conv1,
		depth_radius = 5,
		bias = 2,
		alpha = 0.0001,
		beta = 0.75)
	pool1 = tf.layers.max_pooling2d(inputs = lrn1, pool_size = [2, 2], strides = [2, 2])
	conv2 = tf.layers.conv2d(
		inputs = pool1, 
		filters = 64, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu)
	lrn2 = tf.nn.lrn(
		input = conv2,
		depth_radius = 5,
		bias = 2,
		alpha = 0.0001,
		beta = 0.75)
	pool2 = tf.layers.max_pooling2d(inputs = lrn2, pool_size = [2, 2], strides = [2, 2])
	conv3 = tf.layers.conv2d(
		inputs = pool2, 
		filters = 128, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu)
	conv4 = tf.layers.conv2d(
		inputs = conv3, 
		filters = 128, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu)
	conv5 = tf.layers.conv2d(
		inputs = conv4, 
		filters = 64, 
		kernel_size = [3, 3], 
		padding = 'same',
		activation = tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(inputs = conv5, pool_size = [2, 2], strides = [2, 2])
	pool3_flat = tf.reshape(pool3, [-1, 1536])
	fc1 = tf.layers.dense(inputs = pool3_flat, units=1024, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(inputs = fc1, rate = 0.5, training = mode == tf.estimator.ModeKeys.TRAIN)
	fc2 = tf.layers.dense(inputs = dropout1, units=1024, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(inputs = fc2, rate = 0.5, training = mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs = dropout2, units=10, activation=tf.nn.relu)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input = logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)


def parser(serialized_example):
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
		})
	image = tf.decode_raw(features['image'], tf.uint8)
	image.set_shape([DEPTH * HEIGHT * WIDTH])
	# Reshape from [depth * height * width] to [depth, height, width].
	image = tf.cast(tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),tf.float32)
	label = tf.cast(features['label'], tf.int32)
	return image, label


def my_input_fn(data_file, shuffle=False, repeat=False):
	dataset = tf.data.TFRecordDataset(data_file)
	dataset = dataset.map(parser)
	if shuffle:
		dataset = dataset.shuffle(SHUFFLE_BUFFER)
	dataset = dataset.batch(BATCH_SIZE)
	if repeat:
		dataset = dataset.repeat(NUM_EPOCHS)
	iterator = dataset.make_initializable_iterator()
	images, labels = iterator.get_next()
	return images, lables


def main(unused_argv):
	train_file = tf.train.match_filenames_once("../cifar-10-data/train.tfrecords")
	validation_file = tf.train.match_filenames_once("../cifar-10-data/validation.tfrecords")
	eval_file = tf.train.match_filenames_once("../cifar-10-data/eval.tfrecords")

	alex_classifier = tf.estimator.Estimator(model_fn = alex_net, model_dir="./alex_model")

	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	alex_classifier.train(input_fn=lambda: my_input_fn(train_file), steps=1, hooks=[logging_hook])
	print('done')

	results = alex_classifier.evaluate(input_fn=lambda: my_input_fn(eval_file), steps=1)
	print(results)


if __name__ == "__main__":
	tf.app.run()




	
	














