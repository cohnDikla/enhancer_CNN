__author__ = 'diklac03'

"""
run with python2.7 !!!
create samples of negative data and k-shuflle of them, for all 5 species, and for all values of k,
from k=1 to the MAXIMAL_K value.
"""

import numpy as np
import os
import sys
import math
import ushuffle
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path[:-len('/TF_vs_k_shuffle')]+'/CNN/')
import data_handle
from SampleObject import SampleObject



SAMPLE_LENGTH = 500
BASIS_NUMBER = 4
NUMBER_OF_POSITIVE_EXAMPLES = 12000
MAXIMAL_K = 9

sections = ['train', 'validation', 'test']


train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 1 - train_ratio - validation_ratio

samples_out_base_dir_for_npy_files = "/cs/cbio/dikla/projects/TF_vs_k_shuffle/data/npy_files/"

samples_out_base_dir_for_text_files =  "/cs/cbio/dikla/projects/TF_vs_k_shuffle/data/samples/"

samples_input_base_dir = samples_out_base_dir_for_text_files



output_k_lets_dirs = ["preserving_"+str(k)+"-let_counts/" for k in range(1, MAXIMAL_K+1)]

TF_species_names_ordered = ["Canis_familiaris", "Gallus_gallus", "Homo_sapiens",
                            "Monodelphis_domestica", "Mus_musculus",
                            "All_species_60000", "All_species_12000"]

labels_map = {0: (1, 0), 1: (0, 1)}

# species_names_map = {"Canis_familiaris": "Dog", "Homo_sapiens": "Human",
#                                 "Monodelphis_domestica": "Opossum", "Mus_musculus": "Mouse"}


bases_map = {"A": 0, "C": 1, "G": 2, "T": 3}

def create_directories():
    for i in range(len(TF_species_names_ordered)):
        species_name = TF_species_names_ordered[i]
        species_dir_npy = os.path.join(samples_out_base_dir_for_npy_files, species_name)
        species_dir_text = os.path.join(samples_out_base_dir_for_text_files, species_name)
        for species_dir in [species_dir_npy, species_dir_text]:
            if not os.path.exists(species_dir) and not os.path.isdir(species_dir):
                print("make directory: ", species_dir)
                os.makedirs(species_dir)
            for k_let_dir in output_k_lets_dirs:
                dir_k = os.path.join(species_dir, k_let_dir)
                if not os.path.exists(dir_k) and not os.path.isdir(dir_k):
                    print("make directory: ", dir_k)
                    os.makedirs(dir_k)


def get_base(bases_binary_array):
    # print("array = ",bases_binary_array)
    # print ("type(bases_binary_array) = ", type(bases_binary_array))
    array_list = list(bases_binary_array[0])
    # print("array = ",array_list)
    # print ("type(bases_binary_array) = ", type(array_list))
    indices_bases_map = dict()
    for base,index in bases_map.items():
        indices_bases_map[index] = base
    index = array_list.index(1)
    return indices_bases_map[index]


def translate_bases_from_original_sample(letter):
    if letter in bases_map.keys():
        return bases_map[letter]
    # if the letter is not A,G,C or T, it is N.
    # in this case we choose one of the 4 letters A,C,G,T randomly.
    else:
        print "Error! base is : ", letter
        exit()


def convert_sample_to_matrix(sample):
    """
    This function converts a sample in a string of bases format into a binary matrix with 4 rows
    and SAMPLE_LENGTH=500 columns ('one-hot' matrix).
    """
    idxs = list(map(translate_bases_from_original_sample, sample))

    one_hot_matrix = np.zeros((BASIS_NUMBER, len(sample)))
    one_hot_matrix[idxs, np.arange(len(sample))] = 1
    return one_hot_matrix



def concatenate_bases(bases):
    motif = ""
    for base in bases:
        motif += base
    return motif


def create_one_positive_sample(sample):
    bases = []
    for i in range(SAMPLE_LENGTH):
        bases.append(sample[i])
    # print("bases: ",bases)
    sample_str = concatenate_bases(bases)
    sample_matrix = convert_sample_to_matrix(sample_str)
    # print ("in create_one_positive_sample,  len(sample_matrix) = ", len(sample_matrix))
    # print("sample: ",sample)
    # print ("len(sample) = ",len(sample))
    return sample_str, sample_matrix


####################################### creation of negative examples #########################################

# uses the uShuffle library.
# link - uShuffle:
# http://digital.cs.usu.edu/~mjiang/ushuffle/
# How to Use uShuffle in Python:
# http://digital.cs.usu.edu/~mjiang/ushuffle/python.html

# new creation of negative samples:
# generate the negative samples from the positive samples shuffled,
# while preserving the k-let counts
def create_one_negative_sample_preserve_singles_distribution(sample, k):
    bases = []
    for i in range(SAMPLE_LENGTH):
        bases.append(sample[i])

    sample = concatenate_bases(bases)
    # print "before shuffle = sample: ",sample

    # shuffle the string sequence while preserving the k-let counts:
    sample_str = ushuffle.shuffle(sample, SAMPLE_LENGTH, k)

    # print "after shuffle = sample: ",sample_str

    sample_matrix = convert_sample_to_matrix(sample_str)
    return sample_str, sample_matrix


def create_positive_or_negative_samples(species, is_positive, k=None):
    # print("in create_negative_samples, species_index = ", species_index)
    samples = []
    if is_positive:
        out_path = os.path.join(samples_out_base_dir_for_text_files, species, "positive_samples")
    else:
        out_path = os.path.join(samples_out_base_dir_for_text_files, species, output_k_lets_dirs[k - 1], "negative_samples")
    samples_file_path = os.path.join(samples_input_base_dir, species, species+"_positive_samples.txt")
    with open(out_path, 'w+') as out:
        with open(samples_file_path) as samples_file:
            samples_counter = 0

            for line in samples_file:
                sequence = line
                # print("len(sequence) = ", len(sequence))

                if len(sequence) > SAMPLE_LENGTH:
                    # print ("here")
                    center = int(len(sequence)/2)
                    half_sample = int((SAMPLE_LENGTH+1)/2)

                    sample = sequence[(center-half_sample):(center+half_sample)]
                    # print ("len(new_sample) = ", len(sample))

                    if len(sample) < SAMPLE_LENGTH:
                        sample = sequence[(center-half_sample-1):(center+half_sample+1)]

                elif len(sequence) < SAMPLE_LENGTH:
                    print "error! sequence is too short: len(sequence) = ", len(sequence)

                if is_positive:
                    sample_str, sample_matrix = create_one_positive_sample(sample)
                else:
                    sample_str, sample_matrix = create_one_negative_sample_preserve_singles_distribution(sample, k)

                out.write(sample_str+"\n")

                # sample_matrix = convert_sample_to_matrix(sample)
                # print ("in create_negative_samples,  len(sample_matrix) = ", len(sample_matrix))

                samples.append(sample_matrix)
                # print("added one sample")

                samples_counter += 1

                if samples_counter >= NUMBER_OF_POSITIVE_EXAMPLES:
                    break

    return samples




def create_data(k, species):

    positive_samples = create_positive_or_negative_samples(species, True)
    negative_samples = create_positive_or_negative_samples(species, False, k)
    print "len(positive_samples) = ", len(positive_samples)
    print "len(negative_samples) = ", len(negative_samples)
    return positive_samples, negative_samples


def convert_labels_to_one_hot(raw_labels):
    labels = []
    label2one_hot = {0: (1, 0), 1: (0, 1)}
    for n in range(0, len(raw_labels)):
        labels.append(label2one_hot[raw_labels[n]])
    return labels


def main():
    create_directories()

    # the k-lets counts are preserved during the negative samples generation
    for k in range(1, MAXIMAL_K+1):
        print "start creating data, k = ", k
        for species_name in TF_species_names_ordered:
            if "All" in species_name:
                continue
            print "species_name = ", species_name
            positive_samples, negative_samples = create_data(k, species_name)
            all_Xs = np.array(positive_samples + negative_samples)
            all_ys = np.array([1] * len(positive_samples) + [0] * len(negative_samples))
            perm = np.random.permutation(len(all_Xs))

            all_Xs_shuffled = all_Xs[perm]
            all_ys_shuffled = all_ys[perm]

            samples = np.array(all_Xs_shuffled)
            labels = np.array(convert_labels_to_one_hot(all_ys_shuffled))

            all_samples = []
            for i in range(len(all_Xs_shuffled)):
                sample_matrix = all_Xs_shuffled[i]
                label= all_ys_shuffled[i]
                label_matrix = labels_map[label]
                sample_object = SampleObject(sample_matrix, label_matrix, is_matrix=True)
                all_samples.append(sample_object)


            indices = dict()
            train_start_idx = 0
            train_end_idx = int(math.ceil(len(all_Xs) * train_ratio))
            indices["train"] = (train_start_idx, train_end_idx)
            validation_start_idx = train_end_idx
            validation_end_idx = train_end_idx + int(math.ceil(len(all_Xs) * validation_ratio))
            indices["validation"] = (validation_start_idx, validation_end_idx)
            test_start_idx = validation_end_idx
            test_end_idx = validation_end_idx + int(math.ceil(len(all_Xs) * test_ratio))
            indices["test"] = (test_start_idx, test_end_idx)

            print
            "train_start_idx = ", train_start_idx, \
            "\ntrain_end_idx = ", train_end_idx, \
            "\nvalidation_start_idx = ", validation_start_idx, \
            "\nvalidation_end_idx = ", validation_end_idx, \
            "\ntest_start_idx = ", test_start_idx, \
            "\ntest_end_idx = ", test_end_idx, "\n"

            # save npy files for each species

            path_out_train_X = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                            'train_X')
            path_out_train_y = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                            'train_Y')
            path_out_validation_X = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                                 'validation_X')
            path_out_validation_y = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                                 'validation_Y')
            path_out_test_X = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                           'test_X')
            path_out_test_y = os.path.join(samples_out_base_dir_for_npy_files, species_name, output_k_lets_dirs[k - 1],
                                           'test_Y')

            np.save(path_out_train_X, samples[train_start_idx: train_end_idx])
            np.save(path_out_train_y, labels[train_start_idx: train_end_idx])

            np.save(path_out_validation_X, samples[validation_start_idx: validation_end_idx])
            np.save(path_out_validation_y, labels[validation_start_idx: validation_end_idx])

            np.save(path_out_test_X, samples[test_start_idx: test_end_idx])
            np.save(path_out_test_y, labels[test_start_idx: test_end_idx])

            # write positive and negative text files:
            dir_path = os.path.join(samples_out_base_dir_for_text_files, species_name,
                                    output_k_lets_dirs[k - 1])

            all_Xs_text_shuffled = []
            all_Ys_text_shuffled = []
            for sample in all_samples:
                all_Xs_text_shuffled.append(sample.get_sample_str())
                all_Ys_text_shuffled.append(str(sample.get_label()))

            text_samples = np.array(all_Xs_text_shuffled)
            text_labels = np.array(all_Ys_text_shuffled)

            for section in sections:
                print "section: ", section
                path_out_text_X, path_out_text_Y = data_handle.get_path(dir_path,
                                                                        section)

                start, end = indices[section]
                # print("start, end =", start, end)

                with open(path_out_text_X, 'w') as out_text_samples:
                    string = '\n'.join(text_samples[start: end])
                    out_text_samples.write(string)
                with open(path_out_text_Y, 'w') as out_text_labels:
                    string = '\n'.join(text_labels[start: end])
                    out_text_labels.write(string)


    print("end! :)")

if __name__ == "__main__":
    main()
