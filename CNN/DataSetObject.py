import numpy as np
from SampleObject import SampleObject
from random import shuffle
import math

class DataSetObject:

    def __init__(self, samples_path, labels_path):

        sample_matrices = np.load(samples_path)
        label_matrices = np.load(labels_path)
        self._num_samples = len(sample_matrices)

        self._samples = []
        for index in range(self._num_samples):
            self._samples.append(SampleObject(sample_matrices[index], label_matrices[index],
                                  is_matrix=True))
        self._current_epoch = 1
        self._current_position_in_epoch = 0
        self._batch_size = 0

    def get_samples(self):
        return self._samples

    def get_num_samples(self):
        return self._num_samples

    def get_current_epoch(self):
        return self._current_epoch

    def get_current_position_in_epoch(self):
        return self._current_position_in_epoch

    def get_sample_by_index(self, index):
        return self._samples[index]

    def get_next_batch(self, batch_size=-1):
        """
        Generates the next batch of samples and labels.
        """
        if batch_size > 0:
            self._batch_size = batch_size
        else:
            self._batch_size = self._num_samples
        start = self._current_position_in_epoch
        self._current_position_in_epoch += batch_size

        if self._current_position_in_epoch < self._num_samples:  # epoch is not completed
            batch_matrices = self.create_batch_matrices(start)
            return batch_matrices

        else:  # last iteration of epoch
            difference = self._current_position_in_epoch - self._num_samples
            self._current_position_in_epoch -= difference
            batch_matrices = self.create_batch_matrices(start)

            # start new epoch:
            self._current_position_in_epoch = 0
            shuffle(self._samples)  # Shuffle the data before the next epoch
            self._current_epoch += 1
            return batch_matrices  # list of tuples

    def create_batch_matrices(self, start, end=None):
        if not end:
            end = self._current_position_in_epoch
        batch = self._samples[start: end]
        batch_matrices = [(sample.get_sample_matrix(), sample.get_label_matrix())
                          for sample in batch]
        return batch_matrices

    def get_num_iterations_in_epoch(self):
        return math.ceil(self._num_samples / self._batch_size)

    def initialize_epoch_and_position(self):
        self._current_epoch = 1
        self._current_position_in_epoch = 0

    def get_samples_labels(self, batch):
        batch_samples = []
        batch_labels = []
        for item in batch:
            batch_samples.append(item[0])
            batch_labels.append(item[1])
        return batch_samples, batch_labels









