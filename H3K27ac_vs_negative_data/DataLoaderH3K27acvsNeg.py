"""
Creates the H3K27ac data vs. the new negative non-enhancers data, for all species.
Sets a fixed partition to train, validation and test data for each species.
write all created data samples and labels both as text files and as numpy binary files.

For creating the data of 238000 samples from all species:
the train data is the union of all train data of each species,
the validation data is the union of all train validation of each species,
and the test data is the union of all test data of each species.

For creating the data of 14000 samples from all species:
the train data ia a random 1/17 of the samples from the training data of each species,
the validation data ia a random 1/17 of the samples from the validation data of each species,
and the train data ia a random 1/17 of the samples from the test data of each species.

** Note: it is actually 0.6*238000 samples from all species, i.e. 142800 samples instead of 238000.

* takes random 14000 samples from the positive (H3K27ac) samples
(and not the first 14000 samples).
The negative samples are already randomly chosen from the genome of each species,
such that they are not intersected with any gene (+-15000 bases in each direction)
and with any peak of H3K27ac.

run with python2.7 !!!
"""
import os
import sys
import random
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
base_path = base_path[:-len("/H3K27ac_vs_negative_data")]
sys.path.insert(0, base_path+'/CNN/')
from SampleObject import SampleObject
import data_handle
sys.path.insert(0, base_path+'/simulated_data/')
from DataLoader import DataLoader
from random import shuffle
import numpy as np



class DataLoaderH3K27acvsNeg(DataLoader):
    """
    Data Loader class for the project of H3K27ac_vs_negative_data.
    """
    def __init__(self, project):
        self.ratio_of_samples_from_all_species = 0.6
        # self.ratio_of_samples_from_all_species = 0.8
        # self.ratio_of_samples_from_all_species = 1.0
        super(DataLoaderH3K27acvsNeg, self).__init__(project)


    def get_all_positive_and_negative_samples(self):
        positive_samples_files = []
        original_samples_dir = os.path.join(base_path, "H3K27ac_vs_k_shuffle", "data", "samples")
        for species_name in self.project.species:
            species_dir = os.path.join(original_samples_dir, species_name)
            samples_path = os.path.join(species_dir, species_name + "_positive_samples.txt")
            positive_samples_files.append(samples_path)
        negative_samples_files = []
        for species_name in self.project.species:
            species_dir = os.path.join(original_samples_dir, species_name)
            samples_path = os.path.join(species_dir, species_name + "_new_negative_samples.txt")
            negative_samples_files.append(samples_path)
        self.positive_samples_files = positive_samples_files
        self.negative_samples_files = negative_samples_files


    def create_data_for_each_species(self):
        # create data for each species separately
        for i in range(len(self.project.species)-2):
            species_name = self.project.species[i]
            print "\n\n"
            print "start creating data for species : ", self.project.species[i]
            all_samples =  data_handle.create_data_one_species(self.project, i,
                                                               self.positive_samples_files,
                                                               self.negative_samples_files)
            self.shuffle_and_write_samples(all_samples, species_name)


    def create_data_from_all_species_together(self):
        """
        creating data for all species together -  14000 samples:
        takes random 1/17 of the samples from the training data of each species,
        random 1/17 of the samples from the validation data of each species,
        and random 1/17 of the samples from the test data of each species.
        (same random indices from all species).
        :return:
        """
        print "start creating data for : ", self.project.species[-1], \
              " and : ", self.project.species[-2]
        train_samples_238000 = []
        validation_samples_238000 = []
        test_samples_238000 = []
        train_samples_14000 = []
        validation_samples_14000 = []
        test_samples_14000 = []
        # collect all training, validation and test data from all species:
        section_index = 0
        for section_name in self.sections:
            for j in range(len(self.project.species)-2):
                species_name = self.project.species[j]
                section_samples_per_species = []
                section_file_path = os.path.join(self.project.text_samples_base_dir,
                                                 species_name,
                                                 section_name + "_data")
                with open(section_file_path) as one_species_section_file:
                    line_counter = 0
                    for line in one_species_section_file:
                        split_line = line.split("\t")
                        label = split_line[0]
                        label_int = 1 if label=="True" else 0
                        sequence = (split_line[1])[:-1]  # without \n at the end of the line
                        sample_object = SampleObject(sequence, label_int)
                        section_samples_per_species.append(sample_object)
                        line_counter += 1
                        number_of_samples_in_section = \
                            2*(self.project.get_number_of_samples())*(self.section_ratios[section_index])
                        # TODO for now - takes only <self.ratio_of_samples_from_all_species> samples from each section
                        if line_counter >= (number_of_samples_in_section * self.ratio_of_samples_from_all_species):
                            break

                number_section_samples = line_counter
                # takes 1/17 from each species samples
                random_indices_of_section_samples = random.sample(xrange(number_section_samples),
                                                                  number_section_samples / (len(self.project.species)-2))
                random_section_samples = [section_samples_per_species[j] for j in
                                          range(len(section_samples_per_species))
                                          if j in random_indices_of_section_samples]
                if section_name == 'train':
                    train_samples_14000.extend(random_section_samples)
                    train_samples_238000.extend(section_samples_per_species)
                elif section_name == 'validation':
                    validation_samples_14000.extend(random_section_samples)
                    validation_samples_238000.extend(section_samples_per_species)
                elif section_name == 'test':
                    test_samples_14000.extend(random_section_samples)
                    test_samples_238000.extend(section_samples_per_species)
            section_index += 1
        # print "len(train_samples_14000) = ", len(train_samples_14000)
        # print "len(validation_samples_14000) = ", len(validation_samples_14000)
        # print "len(test_samples_14000) = ", len(test_samples_14000)
        # print "len(train_samples_238000) = ", len(train_samples_238000)
        # print "len(validation_samples_238000) = ", len(validation_samples_238000)
        # print "len(test_samples_238000) = ", len(test_samples_238000)
        # print "total 238000 train+validation+test: ", len(train_samples_238000) \
        #         + len(validation_samples_238000) + len(test_samples_238000)
        # print "total 14000 train+validation+test: ", len(train_samples_14000) \
        #         + len(validation_samples_14000) + len(test_samples_14000)
        for i in range(len(self.project.species)-2, len(self.project.species)):
            species_name = self.project.species[i]
            species_dir_text = os.path.join(self.project.text_samples_base_dir, species_name)
            pos_out_path = os.path.join(species_dir_text, "positive_samples")
            neg_out_path = os.path.join(species_dir_text, "negative_samples")
            species_dir_npy = os.path.join(self.project.samples_base_dir, species_name)
            if i == len(self.project.species)-2:  # iteration #17 - creating data for all species - 238000 samples
                train_samples = train_samples_238000
                validation_samples = validation_samples_238000
                test_samples = test_samples_238000
            elif i == len(self.project.species)-1:  # iteration #18 - creating data for all species - 14000 samples
                train_samples = train_samples_14000
                validation_samples = validation_samples_14000
                test_samples = test_samples_14000
            with open(pos_out_path, 'w') as out_pos:
                with open(neg_out_path, 'w') as out_neg:
                    for section_name in self.sections:
                        print "start section: ", section_name
                        if section_name == 'train':
                            section_samples = train_samples
                        elif section_name == 'validation':
                            section_samples = validation_samples
                        elif section_name == 'test':
                            section_samples = test_samples
                        # write the section samples and labels to text file and numpy file
                        path_out_text_X, path_out_text_Y = data_handle.get_path(species_dir_text,
                                                                                section_name)
                        path_out_npy_X, path_out_npy_Y = data_handle.get_path(species_dir_npy,
                                                                              section_name)
                        shuffle(section_samples)
                        section_sample_matrices = []
                        section_label_matrices = []
                        section_samples_str = []
                        section_labels_str = []
                        for sample in section_samples:
                            section_sample_matrices.append(sample.get_sample_matrix())
                            section_label_matrices.append(sample.get_label_matrix())
                            section_samples_str.append(sample.get_sample_str())
                            section_labels_str.append(str(sample.get_label()))
                        np.save(path_out_npy_X, section_sample_matrices)
                        np.save(path_out_npy_Y, section_label_matrices)
                        with open(path_out_text_X, 'w') as out_text_samples:
                            string = '\n'.join(section_samples_str)
                            out_text_samples.write(string)
                        with open(path_out_text_Y, 'w') as out_text_labels:
                            string = '\n'.join(section_labels_str)
                            out_text_labels.write(string)
                        self.write_positive_and_negative_text_samples_files(section_labels_str,
                                                                            section_samples_str,
                                                                            out_pos, out_neg)

    def write_positive_and_negative_text_samples_files(self, section_labels_str,
                                                       section_samples_str,
                                                       out_pos, out_neg):
        sample_index = 0
        for label in section_labels_str:
            if label == "1":
                out_pos.write(section_samples_str[sample_index]+"\n")
            elif label == "0":
                out_neg.write(section_samples_str[sample_index] + "\n")
            else:
                print("wrong label, exit")
                exit(1)
            sample_index += 1

