"""
A script for visualization of tensors,
especially filters of the first convolutional layer,
that can be presented as motif logo images.
"""
__author__ = 'Dikla Cohn'
import tensorflow as tf
import numpy as np
import os
import math
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import sys
import test_CNN
from subprocess import call
projects_base_path = base_path[:-len('/CNN')]
motifs_path = sys.path.insert(0, projects_base_path+'/motifs/')

trained_on_all_species_only = False

POWER_BASE = 10
script_path = os.path.join(motifs_path, "makefig.pl")
script_create_images = script_path + " -nonumbers"


def export_tensors_as_images(project, best_model_validation_id, tensor_name,
                             filters_dir, species_name, layer_num):
    checkpoints_folder = project.checkpoints_folder_tmp
    # model checkpoint:
    model_variables_path = os.path.join(checkpoints_folder, best_model_validation_id,
                                        best_model_validation_id)
    print("model_def_path : ", model_variables_path)
    # print("in export_tensors_as_images : output_folder = ", output_folder)
    # print("species_name = ", species_name)
    # with open(output_folder+motif_name+"_filters_"+species_name, 'w+') as output_filters:
    output_path_filters = os.path.join(filters_dir, "layer_"+str(layer_num),
                                       "all_filters_" + species_name + ".txt")
    with open(output_path_filters, 'w+') as output_filters:
        filters = []
        try:
            reader = tf.train.NewCheckpointReader(model_variables_path)
            tensor = reader.get_tensor(tensor_name)
            if len(tensor.shape) < 4:
                print("Expecting a 4-d shaped tensor. Got: " + str(tensor.shape))
                exit(-1)
            (height, width, depth, num_of_filters) = tensor.shape
            # print("type(tensor) = ", type(tensor))
            print("height, width, depth, num_of_filters = ", (height, width, depth, num_of_filters))
            # if depth > 1:
            #     print("currently no support for 3-d convolutions")
            #     exit(-1)
            for n in range(num_of_filters):
                # convert tensor to 2-dimensional array
                if layer_num == 1:
                    curr_filter = tensor[:, :, :, n].reshape([height, width])
                    # first_dimension = height
                    # second_dimension = width
                else:
                    curr_filter = tensor[:, :, :, n].reshape([width, depth])
                    # first_dimension = width
                    # second_dimension = depth
                curr_filter = curr_filter.astype(float)
                # print("curr_filter = ",curr_filter)
                output_filters.write(str(curr_filter) + "\n\n")
                if layer_num ==1 :
                    max_value = curr_filter.max() # find the maximal value in the current filter
                    # print("max_value = ", max_value)
                    # divide the current filter by max_value
                    new_curr_filter = np.divide(curr_filter, max_value) # The quotient x1/x2, element-wise.
                    # print("new_curr_filter = ", new_curr_filter)
                else:
                    new_curr_filter = curr_filter
                filters.append(new_curr_filter)
        except Exception as e:
            print(str(e))
        print_each_filter_text_file(filters, layer_num, output_filters, filters_dir,
                                    species_name)
    return filters


def exponent_and_normalize_filters(filters, output_filters, filters_dir, layer_num,
                                   species_name):
    first_dimension = filters[0].shape[0]
    second_dimension = filters[0].shape[1]
    num_filter = -1
    for filter in filters:
        num_filter += 1
        output_filters.write("filter #" + str(num_filter + 1) + ":\n")
        # print("species_name = ", species_name)
        file_path = os.path.join(filters_dir, "layer_"+str(layer_num),
                                 "filter" + str(num_filter + 1) +
                                 "_power_normalize_" + species_name + ".txt")
        filter_values = []
        # print("second dimension = ", second_dimension)
        sum_per_position = [0 for i in range(second_dimension)]
        row_index = 0
        for row in filter:
            row_vector_values = []
            col_index = 0
            for col_value in row:
                power_value = math.pow(POWER_BASE, col_value)
                row_vector_values.append(power_value)
                sum_per_position[col_index] += power_value
                col_index += 1
            row_index += 1
            filter_values.append(row_vector_values)
            output_filters.write("\n\n")
        # normalize the filter values so that the sum in each position will be 1
        new_sum_per_position = [0 for i in range(len(filter[0]))]
        # print("new_sum_per_position = ", new_sum_per_position)
        with open(file_path, 'w') as filter_file:
            # print("opened filter_file: ", file_path)
            for row_index in range(first_dimension):
                row_vector_values = filter_values[row_index]
                col_index = 0
                for col_value in row_vector_values:
                    normalized_value = col_value / sum_per_position[col_index]
                    new_sum_per_position[col_index] += normalized_value
                    filter_file.write(str(normalized_value) + "\t")
                    col_index += 1
                filter_file.write("\n")


def print_each_filter_text_file(filters, layer_num, output_filters, filters_dir, species_name):
    first_dimension = filters[0].shape[0]
    num_filter = -1
    for filter in filters:
        num_filter += 1
        output_filters.write("filter #" + str(num_filter + 1) + ":\n")
        # print("species_name = ", species_name)
        file_path = os.path.join(filters_dir, "layer_"+str(layer_num),
                                 "filter" + str(num_filter + 1) + "_" + species_name + ".txt")
        filter_values = []
        # row_index = 0
        for row in filter:
            row_vector_values = []
            # col_index = 0
            for col_value in row:
                row_vector_values.append(col_value)
                # col_index += 1
            # row_index += 1
            filter_values.append(row_vector_values)
        with open(file_path, 'w') as filter_file:
            # print("opened file_path: ", file_path)
            for row_index in range(first_dimension):
                row_vector_values = filter_values[row_index]
                # col_index = 0
                for col_value in row_vector_values:
                    filter_file.write(str(col_value) + "\t")
                    # col_index += 1
                filter_file.write("\n")
    if layer_num == 1:
        exponent_and_normalize_filters(filters, output_filters, filters_dir,
                                       layer_num, species_name)


def create_directories(project, conv_results_dir):
    filters_dir = os.path.join(conv_results_dir, "filters")
    if not os.path.exists(filters_dir) and not os.path.isdir(filters_dir):
        # print("is dir sample1? ", os.path.isdir(sample_dir))
        print("make directory: ", filters_dir)
        os.makedirs(filters_dir)
        for l_num in range(project.CNN_structure.num_conv_layers):
            layer_dir_path = os.path.join(filters_dir, "layer_" + str(l_num + 1))
            print("make directory: ", layer_dir_path)
            os.makedirs(layer_dir_path)
    return filters_dir


def create_logo_images(layer_num, number_of_filters, filters_dir, train_species):
    """
    save the filters as pdf logo images
    :return:
    """
    if layer_num == 1:
        for i in range(number_of_filters):
            output_image_path = os.path.join(filters_dir, "layer_"+str(layer_num),
                                             "filter" + str(i + 1) + "_" + train_species + ".pdf")
            text_filter_path = os.path.join(filters_dir, "layer_"+str(layer_num),
                                            "filter" + str(i + 1) +
                                            "_power_normalize_" + train_species + ".txt")
            script = "cat " + text_filter_path + " | " + script_create_images + " | " + \
                     "convert eps:- " + output_image_path
            call(script, shell=True)


def get_filters(project, train_species, conv_results_dir, best_model_validation_id):
    for layer_num in range(1, project.CNN_structure.num_conv_layers+1):
        filters_dir = create_directories(project, conv_results_dir)
        filter_name = 'conv'+str(layer_num)+'/weights'
        # graph = restore_model_and_get_filters(project, best_model_validation_id)
        filters = export_tensors_as_images(project, best_model_validation_id, filter_name,
                                           filters_dir, train_species, layer_num)
        create_logo_images(layer_num, len(filters), filters_dir, train_species)


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'tensor_visualization.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)
    number_of_species = len(project.species)
    # with open(conv1_results_output_file, 'w') as conv_file:

    for index_train_species in range(len(project.species)):
        if trained_on_all_species_only:
            trained_species_index = number_of_species - 2  # all species 238000 # change if needed
            train_species = project.species[trained_species_index]
            best_model_validation_id = (list(map_model_ids.keys()))[0]
            if index_train_species!=0:  break
        else:
            best_model_validation_id = sorted_models_list[index_train_species]
            train_species = map_model_ids[best_model_validation_id]

        print("start show convolution on species: ", train_species)
        model_dir = test_CNN.create_directories(project, best_model_validation_id)
        conv_results_dir = os.path.join(model_dir, 'convolution_results')
        get_filters(project, train_species, conv_results_dir, best_model_validation_id)

if __name__ == "__main__":
    main()
    print("End!")

