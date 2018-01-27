import os
import sys
import random
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
base_path = base_path[:-len("TF_vs_negative_data")]
sys.path.insert(0, base_path+'/H3K27ac_vs_negative_data')
from DataLoaderH3K27acvsNeg import DataLoaderH3K27acvsNeg
sys.path.insert(0, base_path+'/CNN')
import data_handle
from SampleObject import SampleObject
from random import shuffle
import numpy as np



class DataLoaderTFvsNeg(DataLoaderH3K27acvsNeg):
    """
    Data Loader class for the project of H3K27ac_vs_negative_data.
    """
    species_names_map = {"Canis_familiaris": "Dog", "Homo_sapiens": "Human",
                                "Monodelphis_domestica": "Opossum", "Mus_musculus": "Mouse"}

    def __init__(self, project):
        # self.ratio_of_samples_from_all_species = 0.6
        # self.ratio_of_samples_from_all_species = 0.8
        self.ratio_of_samples_from_all_species = 1.0
        super(DataLoaderTFvsNeg, self).__init__(project)
        self.ratio_of_samples_from_all_species = 1.0
        self.num_times_negative_data_is_taken = 1

    def create_data_for_each_species(self):
        # create data for each species separately
        print("len(self.project.species) = ", len(self.project.species))
        for i in range(len(self.project.species)-2):
            species_name = self.project.species[i]
            if species_name not in self.species_names_map.keys():
                continue
            print "\n\n"
            print "start creating data for species : ", self.project.species[i]
            all_samples =  data_handle.create_data_one_species(self.project, i,
                                                               self.positive_samples_files,
                                                               self.negative_samples_files)
            self.shuffle_and_write_samples(all_samples, species_name)



    def get_all_positive_and_negative_samples(self):
        positive_samples_files = []
        original_negative_samples_dir = os.path.join(base_path, "H3K27ac_vs_k_shuffle", "data", "samples")
        original_positive_samples_dir = os.path.join(base_path, "TF_vs_k_shuffle", "data", "samples")

        for species_name in self.project.species:
            species_dir = os.path.join(original_positive_samples_dir, species_name)
            samples_path = os.path.join(species_dir, "positive_samples")
            positive_samples_files.append(samples_path)
        negative_samples_files = []
        for species_name in self.project.species:
            if not ("All_species" in species_name):
                new_species_name = self.species_names_map[species_name]
            else:
                new_species_name = species_name
            species_dir = os.path.join(original_negative_samples_dir, new_species_name)
            samples_path = os.path.join(species_dir, new_species_name + "_new_negative_samples.txt")
            negative_samples_files.append(samples_path)
        self.positive_samples_files = positive_samples_files
        self.negative_samples_files = negative_samples_files

    def create_data_from_all_species_together(self):
        """
        creating data for all species together -  14000 samples:
        takes random 1/17 of the samples from the training data of each species,
        random 1/17 of the samples from the validation data of each species,
        and random 1/17 of the samples from the test data of each species.
        (same random indices from all species).
        :return:
        """
        print "start creating data for: ", self.project.species[-1], \
              " and ", self.project.species[-2]
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
                sequence_section_file_path = os.path.join(self.project.text_samples_base_dir,
                                                 species_name,
                                                 section_name + "_X")
                label_section_file_path = os.path.join(self.project.text_samples_base_dir,
                                                          species_name,
                                                          section_name + "_Y")
                sequences = []
                with open(sequence_section_file_path) as one_species_sequence_section_file:
                    for line in one_species_sequence_section_file:
                        if line[-1] == "\n":
                            sequence = line[:-1]
                        else:
                            sequence = line
                        sequences.append(sequence)
                labels = []
                with open(label_section_file_path) as one_species_label_section_file:
                    for line in one_species_label_section_file:
                        if line[-1] == "\n":
                            label = line[0]
                        else:
                            label = line
                        labels.append(int(label))
                samples_counter = 0
                for i in range(len(labels)):
                    sample_object = SampleObject(sequences[i], labels[i])
                    section_samples_per_species.append(sample_object)
                    samples_counter += 1
                    number_of_samples_in_section = \
                        2 * (self.project.get_number_of_samples()) * (self.section_ratios[section_index])
                    # TODO for now - takes only <self.ratio_of_samples_from_all_species> samples from each section
                    if samples_counter >= (number_of_samples_in_section * self.ratio_of_samples_from_all_species):
                        break

                number_section_samples = samples_counter
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

