import numpy as np
import cPickle as pickle


class DataHandler:
	""" Handles batches of data and mini batch generation.

	Data must be presented as python pickled files of a dictionary, dictionary must contain the following key,value pairs:
	'data' -> list of data
	'labels' -> labels corresponding to data in batch

	There must also be a python pickled dictionary called batches.meta that contains the following key,value pairs:
	'num_cases_per_batch' -> the number of data points in each batch file
	'label_names' -> the labels in the batches (only needed for number of labels)
	"""

	def __init__(self,batch_location,number_batches,mini_batch_size,one_hot=True):
		""" Initializes the DataHandler

		:param batch_location: location of python pickled batch files
		:param number_batches: the number of batch files not including test batch or batch meta file
		:param mini_batch_size: desired size of mini batches
		:param one_hot: set true to convert labels to one-hot encoding, else set false. Default true
		"""

		self.batch_location = batch_location
		self.number_batches = number_batches
		self.mini_batch_size = mini_batch_size
		self.current_batch = number_batches
		self.current_mini_batch = 0
		self.one_hot = one_hot

		with open(batch_location + '/batches.meta') as f:
			self.meta = pickle.load(f)

		self.train_size = self.meta['num_cases_per_batch']*self.number_batches

		self.num_mini_batches = self.meta['num_cases_per_batch'] / mini_batch_size

		self.num_labels = len(self.meta['label_names'])

		self.init_test_data()

	#Return meta data
	def get_meta(self):
		""" Returns the meta data of batches

		:return: meta
		"""
		return self.meta

	def next_batch(self):
		""" Sets current batch file to the next one
		"""

		self.current_batch = self.current_batch + 1

		if self.current_batch > self.number_batches:
			self.current_batch = 1

		with open(self.batch_location + '/data_batch_' + str(self.current_batch)) as f:
			self.current_batch_data = pickle.load(f)

		self.batch_data = self.current_batch_data['data']
		self.batch_labels = np.array(self.current_batch_data['labels'])

		if self.one_hot:
			oh = np.zeros((len(self.batch_labels),self.num_labels))

			oh[np.arange(len(self.batch_labels)),self.batch_labels] = 1

			self.batch_labels = oh

	def shuffle_batch(self):
		""" Shuffles the data in the current batch
		"""

		ind = np.arange(self.meta['num_cases_per_batch'])

		np.random.shuffle(ind)

		self.batch_data = self.batch_data[ind]
		self.batch_labels = self.batch_labels[ind]

	def get_next_mini_batch(self):
		""" Generates the next mini batch and returns mini batch and labels

		:return mini_batch_data: mini batch data
		:return mini_batch_labels: mini batch labels
		"""
		if self.current_mini_batch == 0:
			self.next_batch()
			self.shuffle_batch()

		start = self.mini_batch_size*self.current_mini_batch
		end = start + self.mini_batch_size
		mini_batch_data = self.batch_data[start:end]
		mini_batch_labels = self.batch_labels[start:end]

		self.current_mini_batch = (self.current_mini_batch + 1) % self.num_mini_batches

		return mini_batch_data, mini_batch_labels

	def init_test_data(self):
		""" Retrieves and stores the test batch data
		"""
		with open(self.batch_location + '/test_batch') as f:
			self.test_batch_data = pickle.load(f)

		self.test_data = self.test_batch_data['data']
		self.test_labels = np.array(self.test_batch_data['labels'])
		self.test_batch = 0

		if self.one_hot:
			oh = np.zeros((len(self.test_labels),self.num_labels))

			oh[np.arange(len(self.test_labels)),self.test_labels] = 1

			self.test_labels = oh

	def get_test_data(self):
		""" Returns the test data and labels

		:return test_data: test data
		:return test_labels: test labels
		"""

		return self.test_data, self.test_labels

	def get_next_mini_test_batch(self):
		""" Generates and returns the next mini batch of test data and labels

		:return mini_batch_data: test mini batch data
		:return mini_batch_labels: test mini batch labels
		"""

		start = self.mini_batch_size*self.test_batch
		end = start + self.mini_batch_size
		mini_batch_data = self.test_data[start:end]
		mini_batch_labels = self.test_labels[start:end]

		self.test_batch = (self.test_batch + 1) % self.num_mini_batches

		return mini_batch_data, mini_batch_labels

	def get_all_train_data(self):
		""" Appends all training data together from all batch files and returns them

		:return train_x: the training data
		:return train_y: the training labels
		"""
		train_x = np.zeros([0,3072])
		train_y = []

		for i in range(self.number_batches):
			with open(self.batch_location + '/data_batch_' + str(i+1)) as df:
				data = pickle.load(df)
				train_x = np.concatenate((train_x,data['data']))
				train_y = train_y + data['labels']

		return train_x, train_y

def test(batch_location,number_batches,mini_batch_size,one_hot=True):
	tester = DataHandler(batch_location,number_batches,mini_batch_size,one_hot)

	d,l = tester.get_next_mini_batch()

	print d
	print l

if __name__ == '__main__':
	import sys
	bl = sys.argv[1]
	nb = int(sys.argv[2])
	mbs = int(sys.argv[3])
	test(bl,nb,mbs)