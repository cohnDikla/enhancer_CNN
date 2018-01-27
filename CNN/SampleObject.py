import numpy as np

class SampleObject:

    bases_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    labels_map = {0: (1, 0), 1: (0, 1)}
    BASES_NUMBER = 4

    def __init__(self, sample, label, is_matrix=False):
        self._size = len(sample[0])
        if not is_matrix:
            self._sample_str = sample
            self._label = label
            self._sample_matrix = self.convert_sample_to_matrix()
            self._label_matrix = self.convert_label_to_matrix()
        else:
            self._sample_matrix = sample
            self._label_matrix = tuple(label)
            self._sample_str = self.convert_matrix_to_sample()
            self._label = self.convert_matrix_to_label()


    def get_sample_str(self):
        return self._sample_str

    def get_sample_matrix(self):
        return self._sample_matrix

    def get_label(self):
        return self._label

    def get_label_matrix(self):
        return self._label_matrix

    def convert_sample_to_matrix(self):
        """
        This function converts a sample in a string of bases format into a
        binary matrix with 4 rows and SAMPLE_LENGTH=500 columns ('one-hot' matrix).
        """
        for base in self._sample_str:
            if base not in self.bases_map.keys():
                print("Error! self._sample_str = ", self._sample_str)
                exit()
        indices = list(map(lambda l: self.bases_map[l], self._sample_str))
        one_hot_matrix = np.zeros((self.BASES_NUMBER, len(self._sample_str)))
        one_hot_matrix[indices, np.arange(len(self._sample_str))] = 1
        return one_hot_matrix

    def convert_label_to_matrix(self):
        return self.labels_map[self._label]

    def convert_matrix_to_sample(self):
        """
        converts a 4xN one-hot array to its {A,C,G,T}^N representation
        """
        num_to_letter = {item[1]: item[0] for item in self.bases_map.items()}
        seq = []
        for col_index in range(self._size):
            curr_col = self._sample_matrix[:,col_index]
            arg_max = np.argmax(curr_col)
            seq.append(num_to_letter[arg_max])
        return ''.join(seq)

    def convert_matrix_to_label(self):
        """
        converts a 1x2 one-hot array of label to its 0 / 1 representation
        """
        tuple_to_label = {item[1]: item[0] for item in self.labels_map.items()}
        return tuple_to_label[self._label_matrix]
