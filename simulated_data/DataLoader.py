__author__ = 'diklac03'

"""
Creates the simulated data of one TF: CEBBA or HNF4A.
Each sample contains a short sequence sampled from the PWM of the TF.
The location of the planted motif is sampled with normal distribution around the center
of each sample.
Writes all created data samples and labels both as text files and as numpy binary files.
"""

import os
import numpy as np
import math
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path[:-len('/simulated_data')]+'/CNN/')
import data_handle
from random import shuffle


class DataLoader(object):
    """
    Basic DataLoader class for the project of simulated data.
    """
    def __init__(self, project, motifs_base_path=None):
        # set mean and standard deviation for the normal distribution of the motif centers
        # in the simulated positive samples
        self.mu = data_handle.SAMPLE_LENGTH/2
        self.sections = ['train', 'validation', 'test']
        self.section_ratios = [0.8, 0.1, 0.1]
        self.project = project
        self.motifs_dir = motifs_base_path

    def create_npy_files(self):
        print("\nstart creating npy files for samples of motif: ", self.project.PWM)
        PWM_path = os.path.join(self.motifs_dir, self.project.PWM+"_pfm_new.txt")
        if self.project.normal_distribution:
            motif_centers = data_handle.generate_random_motif_centers(self.mu, self.project.sigma,
                                                                      self.project.get_number_of_samples())
            data_handle.draw_histogram(motif_centers, self.project, self.mu, self.project.sigma)
            data_handle.draw_normed_histogram(motif_centers, self.project, self.mu, self.project.sigma)
        else:
            motif_centers = [int(data_handle.SAMPLE_LENGTH / 2)] * self.project.get_number_of_samples()

        all_samples = data_handle.create_simulated_data(motif_centers, PWM_path)
        self.shuffle_and_write_samples(all_samples)

    def shuffle_and_write_samples(self, all_samples, species_name=None):
        if species_name:
            path_out_npy_files_dir = os.path.join(self.project.samples_base_dir, species_name)
            path_out_text_samples_dir = os.path.join(self.project.text_samples_base_dir, species_name)
        else:
            path_out_npy_files_dir = self.project.samples_base_dir
            path_out_text_samples_dir = self.project.text_samples_base_dir
        if self.project.sigma:
            print("self.project.sigma = ", self.project.sigma)

            if not os.path.exists(path_out_npy_files_dir) and not os.path.isdir(path_out_npy_files_dir):
                print("make directory: ", path_out_npy_files_dir)
                os.makedirs(path_out_npy_files_dir)
            if not os.path.exists(path_out_text_samples_dir) and not os.path.isdir(path_out_text_samples_dir):
                print("make directory: ", path_out_text_samples_dir)
                os.makedirs(path_out_text_samples_dir)

        positive_samples_file_path = os.path.join(path_out_text_samples_dir, "positive_samples")
        negative_samples_file_path = os.path.join(path_out_text_samples_dir, "negative_samples")


        shuffle(all_samples)  # shuffle all sample objects in place
        all_Xs_matrices_shuffled = []
        all_Ys_matrices_shuffled = []
        all_Xs_text_shuffled = []
        all_Ys_text_shuffled = []
        positive_samples = []
        negative_samples = []
        for sample in all_samples:
            all_Xs_matrices_shuffled.append(sample.get_sample_matrix())
            all_Ys_matrices_shuffled.append(sample.get_label_matrix())
            all_Xs_text_shuffled.append(sample.get_sample_str())
            all_Ys_text_shuffled.append(str(sample.get_label()))
            if sample.get_label() == 1:
                positive_samples.append(sample)
            elif sample.get_label() == 0:
                negative_samples.append(sample)
        # write positive and negative text files:
        print("write positive and negative text files ... ")
        with open(positive_samples_file_path, 'w') as pos_out:
            for sample in positive_samples:
                pos_out.write(sample.get_sample_str()+"\n")

        with open(negative_samples_file_path, 'w') as neg_out:
            for sample in negative_samples:
                neg_out.write(sample.get_sample_str()+"\n")

        samples = np.array(all_Xs_matrices_shuffled)
        labels = np.array(all_Ys_matrices_shuffled)
        text_samples = np.array(all_Xs_text_shuffled)
        text_labels = np.array(all_Ys_text_shuffled)
        indices = dict()
        train_start_idx = 0
        train_end_idx = math.ceil(len(samples)*self.section_ratios[0])
        indices["train"] = (train_start_idx, train_end_idx)
        validation_start_idx = train_end_idx
        validation_end_idx = train_end_idx + math.ceil(len(samples)*self.section_ratios[1])
        indices["validation"] = (validation_start_idx, validation_end_idx)
        test_start_idx = validation_end_idx
        test_end_idx = validation_end_idx + math.ceil(len(samples)*self.section_ratios[2])
        indices["test"] = (test_start_idx, test_end_idx)
        for section in self.sections:
            print "section: ", section
            path_out_text_X, path_out_text_Y = data_handle.get_path(path_out_text_samples_dir,
                                                                    section)
            print("path_out_npy_files_dir: ", path_out_npy_files_dir)
            path_out_npy_X, path_out_npy_Y = data_handle.get_path(path_out_npy_files_dir,
                                                                  section)
            start, end = indices[section]
            # print("start, end =", start, end)
            np.save(path_out_npy_X, samples[start: end])
            np.save(path_out_npy_Y, labels[start: end])
            with open(path_out_text_X, 'w') as out_text_samples:
                string = '\n'.join(text_samples[start: end])
                out_text_samples.write(string)
            with open(path_out_text_Y, 'w') as out_text_labels:
                string = '\n'.join(text_labels[start: end])
                out_text_labels.write(string)


