import os
import numpy as np
import sys
import random
sys.path.insert(0, '/cs/cbio/dikla/projects/CNN/')
from SampleObject import SampleObject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
A module for data handling.
"""
SAMPLE_LENGTH = 500
BASES_NUMBER = 4
PSEUDO_COUNTS = 1/100
bases_map = {"A": 0, "C": 1, "G": 2, "T": 3}
bases = ["A", "C", "G", "T"]
reverse_complement_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
MAXIMAL_K = 9
k_let_dirs = ["preserving_"+str(k)+"-let_counts/" for k in range(1, MAXIMAL_K+1)]


def get_path(base_dir, section, k=None):
    # if "negative_data_vs_k_shuffle" in base_dir:
    #     index_k = base_dir.index("k_shuffle")
    #     k = int(base_dir[index_k])
    #     path_out_section_X = os.path.join(base_dir,
    #                                k_let_dirs[k-1],  section+'_X')
    #     path_out_section_Y = os.path.join(base_dir,
    #                                k_let_dirs[k-1], section+'_Y')
    if k is not None:
        path_out_section_X = os.path.join(base_dir,
                                          k_let_dirs[k - 1], section + '_X')
        path_out_section_Y = os.path.join(base_dir,
                                          k_let_dirs[k - 1], section + '_Y')
    else:
        path_out_section_X = os.path.join(base_dir, section+'_X')
        path_out_section_Y = os.path.join(base_dir, section+'_Y')
    return path_out_section_X, path_out_section_Y


def read_PWM_from_file(PWM_path):
    counts_all_bases = []
    with open(PWM_path) as pwm:
        base_index = 0
        for line in pwm:
            if line.isspace():
                continue
            if line.startswith(">"):
                continue
            split_line = line.split()
            if len(split_line) == BASES_NUMBER:
                counts_one_position = []
                position_index = base_index
                for base_idx in range(BASES_NUMBER):
                    count = float(split_line[base_idx])
                    counts_one_position.append(count)
                counts_all_bases.append(counts_one_position)
            else:
                counts_one_base = []
                for position in range(len(split_line)):
                    count = float(split_line[position])
                    counts_one_base.append(count)
                counts_all_bases.append(counts_one_base)
            base_index += 1
    array_counts_all_bases = np.array(counts_all_bases)
    if len(array_counts_all_bases[0]) == BASES_NUMBER:
        array_counts_all_bases = np.transpose(array_counts_all_bases)
    return array_counts_all_bases


def read_original_JASPAR_PWM_from_file(PWM_path):
    counts_all_bases = []
    with open(PWM_path) as pwm:
        base_index = 0
        for line in pwm:
            if line.startswith(">MA0") or line.isspace():
                continue
            if line.startswith("A") or line.startswith("C") or line.startswith("G") or line.startswith("T"):
                new_line = " ".join(line.split()[1:-1])
            counts_one_base = []
            split_line = new_line.split()
            for position in range(len(split_line)):
                if (split_line[position]).startswith("["):
                    if len(split_line[position]) > 1:
                        count = float(split_line[position][1:])
                    else:
                        continue
                else:
                    count = float(split_line[position])
                counts_one_base.append(count)
            counts_all_bases.append(counts_one_base)
            base_index += 1
    array_counts_all_bases = np.array(counts_all_bases)
    return array_counts_all_bases

def get_reverse_complement(sample):
    """
    Returns the reverse complement of the given string sample
    :param sample: a string of 'A','C','G','T'
    :return: a string which is the reverse complement of the given string
    """
    rev_comp_string = ""
    # loop over sample in reverse order
    for i in range(len(sample)-1, -1, -1):
        base = sample[i]
        # find the complement base
        comp_base = reverse_complement_map[base]
        rev_comp_string += comp_base
    return rev_comp_string


def get_reverse_complement_matrix(motif_matrix):
    """
    Returns the reverse complement of the given string sample
    :param sample: a string of 'A','C','G','T'
    :return: a string which is the reverse complement of the given string
    """
    motif_length = np.shape(motif_matrix)[1]
    new_matrix = np.empty([BASES_NUMBER, motif_length])
    # loop over the motif matrix in reverse order
    for i in range(motif_length-1, -1, -1):
        for base_index in range(BASES_NUMBER):
            value = motif_matrix[base_index, i]
            new_base_index = bases.index(reverse_complement_map[bases[base_index]])
            new_matrix[new_base_index, i] = value
    return new_matrix


def get_base(bases_binary_array):
    """
    Returns the corresponding base of the giving one-hot array (of size [1,4])
    :param bases_binary_array:
    :return:
    """
    array_list = list(bases_binary_array[0])  # returns a one-hot list of size 4
    indices_bases_map = dict()
    for base, index in bases_map.items():
        indices_bases_map[index] = base
    index = array_list.index(1)  # find the index of the '1'
    return indices_bases_map[index]


def concatenate_bases(bases):
    motif = ""
    for base in bases:
        motif += base
    return motif


def create_one_motif(normalized_PWM):
    motif_length = len(normalized_PWM[0])
    bases = []
    for position in range(motif_length):
        array = np.random.multinomial(1, normalized_PWM[:, position], size=1)
        bases.append(get_base(array))
    motif = concatenate_bases(bases)
    return motif


def create_frequency_matrix_with_pseudo_counts(PWM_path, is_original=False, is_denovo=False):
    if is_original:
        PWM = read_original_JASPAR_PWM_from_file(PWM_path)
    else:
        PWM = read_PWM_from_file(PWM_path)
    if not is_denovo:
        sum_counts_per_position = sum(PWM[:, 0])
        normalized_PWM = []
        # normalization:
        for base_index in range(len(PWM)):
            normalized_one_base = [(float(i)/sum_counts_per_position) for i in PWM[base_index]]
            normalized_PWM.append(normalized_one_base)
        array_normalized_PWM = np.array(normalized_PWM)
        new_sum_per_position = sum(array_normalized_PWM[:, 0])
        new_sum_per_position += (BASES_NUMBER*PSEUDO_COUNTS)
        PWM_with_pseudo_counts = []
        # add pseudo counts:
        for base_index in range(len(array_normalized_PWM)):
            one_base_with_pseudo_counts = [float((i+PSEUDO_COUNTS)/new_sum_per_position)
                                           for i in array_normalized_PWM[base_index]]
            PWM_with_pseudo_counts.append(one_base_with_pseudo_counts)
    else:
        PWM_with_pseudo_counts = PWM
    array_PWM_with_pseudo_counts = np.array(PWM_with_pseudo_counts)
    return array_PWM_with_pseudo_counts


def create_frequency_matrix(PWM_path):
    PWM = read_PWM_from_file(PWM_path)
    sum_counts_per_position = sum(PWM[:, 0])
    normalized_PWM = []
    # normalization:
    for base_index in range(len(PWM)):
        normalized_one_base = [(float(i)/sum_counts_per_position) for i in PWM[base_index]]
        normalized_PWM.append(normalized_one_base)
    array_normalized_PWM = np.array(normalized_PWM)
    return array_normalized_PWM


def create_all_motifs(PWM_path, number_of_samples):
    """
    create motifs that will be planted in the positive samples of the simulated data.
    :param PWM_path:
    :param number_of_samples:
    :return:
    """
    motifs = []
    normalized_PWM_array = create_frequency_matrix_with_pseudo_counts(PWM_path)
    for i in range(number_of_samples):
        motif = create_one_motif(normalized_PWM_array)
        motifs.append(motif)
    return motifs


def concatenate_random_bases_and_motif(bases, motif=None, motif_center_index=None):
    sample = ""
    bases_length = len(bases)
    if motif_center_index and motif:
        motif_length = len(motif)
        motif_start_index = motif_center_index - int(motif_length / 2)
        for i in range(motif_start_index):
            sample += bases[i]
        sample += motif
        for i in range(motif_start_index, bases_length):
            sample += bases[i]
    else:
        for i in range(SAMPLE_LENGTH):
            sample += bases[i]
    return sample


def create_one_positive_sample_simulated_data(motif, motif_center_index):
    bases = []
    motif_length = len(motif)
    for i in range(SAMPLE_LENGTH - motif_length):
        # sample bases from uniform distribution
        array = np.random.multinomial(1, [1/float(BASES_NUMBER)] * BASES_NUMBER, size=1)
        bases.append(get_base(array))
    sample = concatenate_random_bases_and_motif(bases, motif, motif_center_index)
    sample_object = SampleObject(sample, 1)
    return sample_object


def create_one_negative_sample_simulated_data():
    bases = []
    for i in range(SAMPLE_LENGTH):
        array = np.random.multinomial(1, [1/4.] * 4, size=1)  # uniform distribution
        bases.append(get_base(array))
    sample = concatenate_random_bases_and_motif(bases)
    sample_object = SampleObject(sample, 0)
    return sample_object


def create_positive_samples_simulated_data(motif_centers, PWM_path):
    number_of_samples = len(motif_centers)
    motifs = create_all_motifs(PWM_path, number_of_samples)
    positive_samples = []
    for i in range(number_of_samples):
        sample_object = create_one_positive_sample_simulated_data(motifs[i], motif_centers[i])
        positive_samples.append(sample_object)
    return positive_samples

def create_positive_or_negative_samples(project, is_positive, index_of_iteration,
                                        positive_samples_files, negative_samples_files,
                                        species_name=None):
    all_samples = []
    i = index_of_iteration
    out_path_dir = project.text_samples_base_dir
    if species_name:
        out_path_dir = os.path.join(out_path_dir,  species_name)
    if not os.path.exists(out_path_dir) and not os.path.isdir(out_path_dir):
        print("make directory: ", out_path_dir)
        os.makedirs(out_path_dir)
    if is_positive:
        out_path_str = os.path.join(out_path_dir, "positive_samples")
        samples_paths = positive_samples_files[i]
    else:
        out_path_str = os.path.join(out_path_dir, "negative_samples")
        samples_paths = negative_samples_files[i]
    if not type(samples_paths) == list:
        samples_paths = [samples_paths]
    number_of_files = len(samples_paths)
    with open(out_path_str, 'w+') as out:
        for file_idx in range(number_of_files):
            samples_path_1 = samples_paths[file_idx]
            samples = []
            with open(samples_path_1) as samples_file:
                # read all the lines from the file once,
                lines = samples_file.readlines()
                # and then select random <project.get_number_of_samples> lines
                # from the saved list of all lines in memory
                samples_counter = 0
                for line in lines:
                    sequence = line[:SAMPLE_LENGTH]  # without \n at the end
                    sample_object = create_one_sample(is_positive, sequence)  # takes the line as sample
                    samples.append(sample_object)
                    if i != len(project.species) - 1:
                        out.write(sequence + "\n")
                    out.write(sequence + "\n")
                    # sample_matrix = sample_object.convert_sample_to_matrix()
                    samples_counter += 1
                    if samples_counter == project.get_number_of_samples():
                        break
            all_samples.extend(samples)

    return all_samples


def create_negative_samples_simulated_data(number_of_samples):
    negative_samples = []
    for i in range(number_of_samples):
        sample_object = create_one_negative_sample_simulated_data()
        negative_samples.append(sample_object)
    return negative_samples


def create_simulated_data(motif_centers, PWM_path):
    positive_samples = create_positive_samples_simulated_data(motif_centers, PWM_path)
    negative_samples = create_negative_samples_simulated_data(len(motif_centers))
    return positive_samples + negative_samples


def create_one_sample(is_positive, sequence):
    label = 1 if is_positive else 0
    sample_object = SampleObject(sequence, label)
    return sample_object


def create_data_one_species(project, index_of_iteration, positive_samples_files,
                            negative_samples_files):
    species_name = project.species[index_of_iteration]
    positive_samples_objects = create_positive_or_negative_samples(project, True, index_of_iteration,
                                                 positive_samples_files, negative_samples_files, species_name)

    negative_samples_objects = create_positive_or_negative_samples(project, False, index_of_iteration,
                                                 positive_samples_files, negative_samples_files, species_name)
    return positive_samples_objects + negative_samples_objects


def generate_random_motif_centers(mu, sigma, number_of_samples):
    # Draw random samples from a normal (Gaussian) distribution.
    motif_centers = np.random.normal(mu, sigma, number_of_samples)

    # while (max(motif_centers) > 300 or min(motif_centers) < 200):
    #     motif_centers = np.random.normal(mu, sigma, number_of_samples)
    motif_centers_int = [int(i) for i in motif_centers]
    return motif_centers_int


def draw_histogram(motif_centers_int, project, mu, sigma):
    samples_base_dir = os.path.join(project.base_dir_data_path,
                                    project.distribution_samples_center_dir,
                                    project.PWM)
    figure_hist_path = os.path.join(samples_base_dir,
                                    "distribution_of_" + project.PWM + "_motif_center_sigma_"
                                    + str(sigma) + ".pdf")
    motif_name = " ".join(project.PWM.split("_"))
    maximum = max(motif_centers_int)
    minimum = min(motif_centers_int)
    # number_of_bins = maximum-minimum+1
    number_of_bins = 100
    bin_numbers = np.linspace(minimum, maximum, number_of_bins)
    fig = plt.figure()
    fig.suptitle("Distribution of "+motif_name+" motif center position\n"
                 "in simulated positive samples", fontsize=14)
    ax = fig.add_subplot(111)

    count, bins, ignored = plt.hist(motif_centers_int, bin_numbers, color="#0000FF")
    max_count = max(count)
    ax.text(50, max_count-int(max_count/7),
            'mean: '+str(int(mu))+'\nstandard deviation: '+str(sigma),
            style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})
    ax.set_xlabel('motif center position')
    ax.set_ylabel('counts')
    ax.set_xlim(0, SAMPLE_LENGTH)
    plt.savefig(figure_hist_path)
    print("saving figure: ", figure_hist_path)


def draw_normed_histogram(motif_centers_int, project, mu, sigma):
    samples_base_dir = os.path.join(project.base_dir_data_path,
                                    project.distribution_samples_center_dir,
                                    project.PWM)
    # normed histogram:
    figure_hist_normed_path = os.path.join(samples_base_dir,
                                           "normed_distribution_of_" + project.PWM + "_motif_center_sigma_"
                                           + str(sigma) + ".pdf")
    motif_name = " ".join(project.PWM.split("_"))
    fig = plt.figure()
    fig.suptitle("Normed distribution of " + motif_name + " motif center position\n"
                                                    "in simulated positive samples",
                                                    fontsize=14)
    ax = fig.add_subplot(111)
    maximum = max(motif_centers_int)
    minimum = min(motif_centers_int)
    number_of_bins = 100
    bin_numbers = np.linspace(minimum, maximum, number_of_bins)
    # normed or is True - therefore the weights are normalized,
    # so that the integral of the density over the range remains 1.
    count, bins, ignored = plt.hist(motif_centers_int, bin_numbers, normed=True,
                                    color="#0000FF", edgecolor="none")
    # The probability density for the Gaussian distribution:
    y = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-((bins-mu)**2)/(2*(sigma ** 2)))
    ax.plot(bins, y, linewidth=2, color='r')
    ax.text(50, 0.008,
            'mean: ' + str(int(mu)) + '\nstandard deviation: ' + str(sigma),
            style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})
    ax.set_xlabel('motif center position')
    ax.set_ylabel('counts')
    ax.set_xlim(0, SAMPLE_LENGTH)
    plt.savefig(figure_hist_normed_path)
    print("saving figure: ", figure_hist_normed_path)

def remove_files(base_dir, extension_to_remove, species):
    for species_name in species:
        species_dir = os.path.join(base_dir, species_name)
        for filename in os.listdir(species_dir):
            if filename.endswith(extension_to_remove):
                path = os.path.join(species_dir, filename)
                os.remove(path)





