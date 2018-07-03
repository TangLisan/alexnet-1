import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import alexnet
import alexnet_train


def main(argv=None):

	# test data input
	test_file = tf.train.match_filenames_once("../cifar-10-data/eval.tfrecords")
	test_dataset = tf.data.TFRecordDataset(test_file)
	test_dataset = test_dataset.map(alexnet.test_parser)
	test_dataset = test_dataset.batch(1000)
	test_dataset = test_dataset.repeat(None)
	test_iterator = test_dataset.make_initializable_iterator()
	test_image, test_label = test_iterator.get_next()

	# test
	test_logits = alexnet.inference(test_image, False)
	test_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_logits, labels=test_label)
	test_loss = tf.reduce_mean(test_cross_entropy)
	correct_prediction = tf.equal(tf.argmax(test_logits, -1, output_type=tf.int32), test_label)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		sess.run(test_iterator.initializer)
		ckpt = tf.train.get_checkpoint_state(alexnet_train.MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			# for i in range(10):
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			test_loss_value, accuracy_score = sess.run([test_loss, accuracy])
			print("steps: %s, test_loss: %g, accuracy: %g" % (global_step, test_loss_value, accuracy_score))
		else:
			print('No checkpoint file found')
			return


if __name__ == '__main__':
	main()








