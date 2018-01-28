import tensorflow as tf
import numpy as np
import os
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import sys
import test_CNN
from DataSetObject import DataSetObject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BATCH_SIZE_TEST = 1
create_figure_for_each_filter = False

trained_on_all_species_only = False
# trained_species_index = 0

class tmptype(object):
    pass


def eval_tensor(op, sample):
    values = []
    matrix_sample = sample.get_sample_matrix()
    matrix_label = sample.get_label_matrix()
    matrices_list = [matrix_sample]
    labels_list = [matrix_label]
    values.extend(op.eval(feed_dict={"input_x:0": matrices_list,
                                     "input_y:0": labels_list,
                                     "input_keep_prob:0": 1}))
    return values


def get_conv_results_all_layers(project, graph, sample):
    conv_results_all_layers = []
    for layer_num in range(project.CNN_structure.num_conv_layers):
        all_convs_one_layer = []
        conv_tensor = graph.get_tensor_by_name("conv" + str(layer_num + 1) + "/relu_conv" +
                                               str(layer_num + 1) + ":0")
        values = eval_tensor(conv_tensor, sample)
        all_convs_one_layer.extend(values)
        conv_results_all_layers.append(all_convs_one_layer)
    return conv_results_all_layers


def restore_model_and_get_conv_results(project, train_species,
                                       best_model_validation_id, argmin, argmax):
    print("start show convolution on species: ", train_species)
    checkpoints_folder = project.checkpoints_folder_tmp
    test_x_path, test_y_path = test_CNN.get_test_samples_path(project, train_species)
    with tf.Session() as sess:
        # load meta graph and restore weights:
        print("best_model_validation_id = ", best_model_validation_id)
        # model checkpoint:
        model_variables_path = os.path.join(checkpoints_folder, best_model_validation_id,
                                            best_model_validation_id)
        # model definition:
        model_def_path = os.path.join(checkpoints_folder, best_model_validation_id,
                                      best_model_validation_id + ".meta")
        print("model_def_path : ", model_def_path)
        saver = tf.train.import_meta_graph(model_def_path)
        saver.restore(sess, model_variables_path)
        graph = tf.get_default_graph()
        test_set = DataSetObject(test_x_path, test_y_path)
        # test_samples = test_set.get_next_batch(BATCH_SIZE_TEST)
        sample_max_score = test_set.get_sample_by_index(argmax)
        sample_min_score = test_set.get_sample_by_index(argmin)
        conv_results_all_layers_max = get_conv_results_all_layers(project, graph,
                                                                  sample_max_score)
        conv_results_all_layers_min = get_conv_results_all_layers(project, graph,
                                                                  sample_min_score)
    # index_of_sample = test_set.get_current_position_in_epoch()
    # test_samples_matrices, test_correct_labels = test_set.get_samples_labels(test_samples)
    return conv_results_all_layers_max, sample_max_score,\
        conv_results_all_layers_min, sample_min_score


def create_directories(project, conv_results_dir, is_max):
    if is_max:
        sample_dir = os.path.join(conv_results_dir, "positive_sample")
    else:
        sample_dir = os.path.join(conv_results_dir, "negative_sample")
    if not os.path.exists(sample_dir) and not os.path.isdir(sample_dir):
        # print("is dir sample1? ", os.path.isdir(sample_dir))
        print("make directory: ", sample_dir)
        os.makedirs(sample_dir)
        for l_num in range(project.CNN_structure.num_conv_layers):
            layer_dir_path = os.path.join(sample_dir, "layer_" + str(l_num + 1))
            print("make directory: ", layer_dir_path)
            os.makedirs(layer_dir_path)
    return sample_dir


def write_text_files(project, layer_num, conv_results_dir,
                     array_all_convs_one_layer, sample, is_max):
    # save results of all filters in the current layer also as text file
    sample_dir = create_directories(project, conv_results_dir, is_max)
    text_path = os.path.join(sample_dir, "layer_" + str(layer_num + 1),
                             "layer_" + str(layer_num + 1) + '_all_filters_conv_results.txt')
    with open(text_path, 'wb') as f:
        np.savetxt(f, array_all_convs_one_layer[0], fmt='%5s', delimiter='\t',
                   newline='\n')
        # np.savetxt(f, array_all_convs_one_layer[0],
        #                        fmt='%s', delimiter='\t', newline='\n',
        #                        header='', footer='', comments='# ')
    sample_out_path = os.path.join(sample_dir, "sample.txt")
    with open(sample_out_path, 'w') as sample_file:
        sample_file.write(sample.get_sample_str() + "\n\n")
        sample_file.write("label:\t" + str(sample.get_label()) + '\n')
    return sample_dir


def draw_figure_for_each_layer(layer_num, array_all_convs_one_layer, sample_dir, is_max):
    # print("layer_num: ", layer_num)
    # print("tensor shape = ", array_all_convs_one_layer.shape)
    num_filters_in_layer = array_all_convs_one_layer[0].shape[2]
    # print("num_filters_in_layer = ", num_filters_in_layer)
    result_width = array_all_convs_one_layer[0].shape[1]
    # print("result_width = ", result_width)

    # <num_filters_in_layer> subplots sharing both x/y axes:
    figure, axes = plt.subplots(num_filters_in_layer, sharex=True, sharey=True)
    for filter_num in range(1, num_filters_in_layer + 1):
        if create_figure_for_each_filter:
            fig, ax1 = plt.subplots()
        filter_conv_results = array_all_convs_one_layer[0][0, :, filter_num-1]
        ind = np.arange(result_width)  # the x locations for the vector width
        max_value = max(filter_conv_results)
        # print("max_value: ", max_value)
        min_value = min(filter_conv_results)
        # print("min_value: ", min_value)
        #  normalize such that all values will be between 0 and 1
        if max_value != 0:
            filter_conv_results = [(i/max_value) for i in filter_conv_results]
        else:
            print("max conv value = 0, filter_num = ", filter_num, ", layer num = ", layer_num+1,
                  ", is posotive sample: ", is_max)
        # print("after normalization:")
        max_value = max(filter_conv_results)
        if max_value is None:
            print("Error")
            exit()
        # print("max_value: ", max_value)
        min_value = min(filter_conv_results)
        # print("min_value: ", min_value)
        if create_figure_for_each_filter:
            rects = ax1.bar(ind, filter_conv_results, edgecolor='b')
            # add some text for labels, title and axes ticks
            ax1.set_title('filter'+str(filter_num)+' convolution results')
            figure_path = os.path.join(sample_dir, "layer_"+str(layer_num+1),
                                       'filter_' + str(filter_num) +'_conv_results.pdf')
            fig.savefig(figure_path, foemat='pdf')
        max_height = 1  # normalization
        axes[filter_num-1].bar(ind, filter_conv_results, edgecolor='b')
        axes[filter_num-1].set_ylim([min_value, 1])
        axes[filter_num-1].get_yaxis().set_ticks([])
        axes[filter_num - 1].text(-(result_width/20), 0.05*max_height,
                                  "filter "+str(filter_num), rotation=0, horizontalalignment='right',
                                  va='bottom')
        axes[filter_num - 1].set_xlim([0, result_width])
        axes[filter_num - 1].get_xaxis().set_data_interval(0, result_width)

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    figure.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in figure.axes[:-1]], visible=False)
    if is_max:
        axes[0].set_title("output convolutions layer"+str(layer_num+1)+", positive sample")
    else:
        axes[0].set_title("output convolutions layer"+str(layer_num+1)+", negative sample")
    figure_path_all_convs_in_layer = os.path.join(sample_dir, "layer_" + str(layer_num + 1),
                               'output_all_convolutions_in_layer_'+str(layer_num + 1)+'.pdf')
    figure.savefig(figure_path_all_convs_in_layer, format='pdf')


def create_convolution_figures_and_text_files(project, best_model_validation_id, train_species,
                                              conv_results_all_layers,
                                              sample, is_max):
    model_dir = test_CNN.create_directories(project, best_model_validation_id)
    conv_results_dir = os.path.join(model_dir, 'convolution_results')
    for layer_num in range(project.CNN_structure.num_conv_layers):
        # print("layer_num = ", layer_num+1)
        array_all_convs_one_layer = np.array(conv_results_all_layers[layer_num])
        sample_dir = write_text_files(project, layer_num, conv_results_dir,
                                      array_all_convs_one_layer, sample, is_max)
        draw_figure_for_each_layer(layer_num, array_all_convs_one_layer, sample_dir, is_max)


def main():

    project = test_CNN.get_project_and_check_arguments(sys.argv, 'show_convolution.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)
    number_of_species = len(project.species)
    if trained_on_all_species_only:
        trained_species_index = number_of_species - 2  # all species 238000
        train_species = project.species[trained_species_index]
        best_model_validation_id = (list(map_model_ids.keys()))[0]
        test_species = train_species
        array_true_labels, array_prediction_scores = test_CNN.import_model_and_test(
            project, best_model_validation_id, test_species, train_species)

        max_score = max(array_prediction_scores)
        print("max_score = ", max_score)
        argmax = np.argmax(array_prediction_scores)
        print("argmax = ", argmax)
        label_max = array_true_labels[argmax]
        print("label_max = ", label_max)
        min_score = min(array_prediction_scores)
        print("min_score = ", min_score)
        argmin = np.argmin(array_prediction_scores)
        print("argmin = ", argmin)
        label_min = array_true_labels[argmin]
        print("label_min = ", label_min)
        conv_results_all_layers_max, sample_max_score, conv_results_all_layers_min, sample_min_score = \
            restore_model_and_get_conv_results(project, train_species,
                                               best_model_validation_id, argmin, argmax)
        create_convolution_figures_and_text_files(project, best_model_validation_id, train_species,
                                                  conv_results_all_layers_max,
                                                  sample_max_score, True)
        create_convolution_figures_and_text_files(project, best_model_validation_id, train_species,
                                                  conv_results_all_layers_min,
                                                  sample_min_score, False)
    else:
        for best_model_validation_id in sorted_models_list:
            train_species = map_model_ids[best_model_validation_id]
            test_species = train_species
            array_true_labels, array_prediction_scores = test_CNN.import_model_and_test(
                project, best_model_validation_id, test_species, train_species)

            max_score = max(array_prediction_scores)
            print("max_score = ", max_score)
            argmax = np.argmax(array_prediction_scores)
            print("argmax = ", argmax)
            label_max = array_true_labels[argmax]
            print("label_max = ", label_max)
            min_score = min(array_prediction_scores)
            print("min_score = ", min_score)
            argmin = np.argmin(array_prediction_scores)
            print("argmin = ", argmin)
            label_min = array_true_labels[argmin]
            print("label_min = ", label_min)
            conv_results_all_layers_max, sample_max_score, conv_results_all_layers_min, sample_min_score =\
                restore_model_and_get_conv_results(project, train_species,
                                                   best_model_validation_id, argmin, argmax)
            create_convolution_figures_and_text_files(project, best_model_validation_id, train_species,
                                                      conv_results_all_layers_max,
                                                      sample_max_score, True)
            create_convolution_figures_and_text_files(project, best_model_validation_id, train_species,
                                                      conv_results_all_layers_min,
                                                      sample_min_score, False)
    print("End!!!")


if __name__ == "__main__":
    main()
