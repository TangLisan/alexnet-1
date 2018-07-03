import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import alexnet


BATCH_SIZE = 100
SHUFFLE_BUFFER = 20000
TRAINING_STEPS = 50000
LEARNING_RATE_BASE = 0.003
DECAY_STEPS = 500
LEARNING_RATE_DECAY = 0.9
MOMENTUM = 0.9
MODEL_SAVE_PATH="model/"
MODEL_NAME="alexnet_model"


def main(argv=None):
	# train data input
	train_file = tf.train.match_filenames_once("../cifar-10-data/train.tfrecords")
	train_dataset = tf.data.TFRecordDataset(train_file)
	train_dataset = train_dataset.map(alexnet.train_parser)
	train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
	train_dataset = train_dataset.repeat(None)
	train_iterator = train_dataset.make_initializable_iterator()
	train_image, train_label = train_iterator.get_next()

	# train
	y = alexnet.inference(train_image, True)
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
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		sess.run(train_iterator.initializer)
		f = open('result/result2.txt', 'a')
		for i in range(TRAINING_STEPS):
			_, loss_value, step, accuracy_score = sess.run([train_step, loss, global_step, accuracy])

			if i % 100 == 0:
				print("steps: %d, loss: %g, accuracy: %g" % (step, loss_value, accuracy_score))
				f.write("%d\t%f\t%f\n" % (step, loss_value, accuracy_score))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
		f.close()


if __name__ == '__main__':	
	main()








