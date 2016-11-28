import tensorflow as tf
import tflearn
import numpy as np
from data import DataHandler
from network_defs import *
import time
import os

tf.python.control_flow_ops = tf


def example_net(x):
	network = tflearn.conv_2d(x, 32, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.conv_2d(network, 64, 3, activation='relu')
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.fully_connected(network, 512, activation='relu')
	network = tflearn.dropout(network, 0.5)
	network = tflearn.fully_connected(network, 10, activation='softmax')

	return network


def trythisnet(x):
	network = tflearn.conv_2d(x,64,5,activation='relu')
	network = tflearn.max_pool_2d(network,3,2)
	network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.conv_2d(network,64,5,activation='relu')
	network = tflearn.local_response_normalization(network,4,alpha=0.001/9.0)
	network = tflearn.max_pool_2d(network,3,2)
	network = tflearn.fully_connected(network,384,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,192,activation='relu',weight_decay=0.004)
	network = tflearn.fully_connected(network,10,activation='softmax',weight_decay=0.0)

	return network

def resnet1(x, n = 5):
	net = tflearn.conv_2d(x, 16, 3, regularizer='L2', weight_decay=0.0001)
	net = tflearn.residual_block(net, n, 16)
	net = tflearn.residual_block(net, 1, 32, downsample=True)
	net = tflearn.residual_block(net, n - 1, 32)
	net = tflearn.residual_block(net, 1, 64, downsample=True)
	net = tflearn.residual_block(net, n - 1, 64)
	net = tflearn.batch_normalization(net)
	net = tflearn.activation(net, 'relu')
	net = tflearn.global_avg_pool(net)
	# Regression
	net = tflearn.fully_connected(net, 10, activation='softmax')

def train_nn_tflearn(data_handler,num_epochs=50):

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

	batch_size = data_handler.mini_batch_size

	img_prep = tflearn.ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	img_aug = tflearn.ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25)

	x = tflearn.input_data(shape=[None, 32, 32, 3], dtype='float', data_preprocessing=img_prep,
						   data_augmentation=img_aug)
	# x = tf.placeholder('float', [None, 32, 32, 3])
	#y = tf.placeholder('float', [None, 10])

	# test_data, test_labels = data_handler.get_test_data()
	# test_data = test_data.reshape([-1,32,32,3])

	ntrain = data_handler.train_size
	ntest = data_handler.meta['num_cases_per_batch']

	# from tflearn.datasets import cifar10
	# (X, Y), (X_test, Y_test) = cifar10.load_data(dirname="/home/hamza/meh/bk_fedora24/Documents/tflearn_example/cifar-10-batches-py")
	# X, Y = tflearn.data_utils.shuffle(X, Y)
	# Y = tflearn.data_utils.to_categorical(Y, 10)
	# Y_test = tflearn.data_utils.to_categorical(Y_test, 10)

	X, Y = data_handler.get_all_train_data()

	X, Y = tflearn.data_utils.shuffle(X, Y)

	X = np.dstack((X[:, :1024], X[:, 1024:2048], X[:, 2048:]))

	X = X/255.0

	X = X.reshape([-1,32,32,3])

	Y = tflearn.data_utils.to_categorical(Y,10)

	X_test, Y_test = data_handler.get_test_data()

	X_test = np.dstack((X_test[:, :1024], X_test[:, 1024:2048], X_test[:, 2048:]))

	X_test = X_test/255.0

	X_test = X_test.reshape([-1,32,32,3])

	#network = tflearn.regression(net3(x),optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)
	mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
	network = tflearn.regression(resnet1(x),optimizer=mom,loss='categorical_crossentropy')

	print np.shape(X)
	print np.shape(Y)
	print network

	model = tflearn.DNN(network,tensorboard_verbose=0)
	model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
			  show_metric=True, batch_size=data_handler.mini_batch_size, run_id='cifar10_cnn')

if __name__ == '__main__':
	import sys

	bl = sys.argv[1]
	nb = int(sys.argv[2])
	mbs = int(sys.argv[3])
	nep = int(sys.argv[4])

	handler = DataHandler(bl,nb,mbs)
	#train_nn(0,handler)
	train_nn_tflearn(handler,nep)
