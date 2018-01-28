__author__ = 'Dikla Cohn'
"""
display_heatmap_TF.py
~~~~~~~~~~~~~~~~~~~~~~
This module is responsible for displaying the test AUC results of all species,
both as a graph with numerical values shown on it, and as a heatmap.
"""
__author__ = 'Dikla Cohn'
import os
import numpy as np
import test_CNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys





# def get_map_model_ids_species(train_file_path):
#     map_model_ids_species = dict()
#     with open(train_file_path) as train_file:
#         for line in train_file:
#             if re.match("^\s$", line):
#                 continue
#             split_line = line.split("\t")
#             train = split_line[0].split()[1]
#             model_id = split_line[2].split()[1]
#             map_model_ids_species[model_id] = train
#     print("map_model_ids_species: ", map_model_ids_species)
#     return map_model_ids_species



# def display_confusion_matrix(data, network, labels, is_training):
#     """
#     This function displays the confusion matrix of the given data.
#     """
#     matrix = np.zeros((len(labels), len(labels)))
#     x_axis = [i for i in range(len(labels))]
#     dict = {}
#     for index, label in enumerate(labels):
#         dict[label] = index
#     successes = 0
#     failures = 0
#     for (x, y) in data:
#         if is_training:
#             y = np.argmax(y)
#         string_y = data_loader.LABELS[y]
#         feedforward_result = np.argmax(network.feedforward(x))
#         predicted_class = labels[feedforward_result]
#         if y == feedforward_result:
#             successes += 1
#         else:
#             failures += 1
#         matrix[dict[string_y]][dict[predicted_class]] += 1
#     print ("successes = ",successes)
#     print ("failures = ",failures)
#     fig, ax = plt.subplots()
#     plt.grid(True)
#     for true_label_index in range(len(matrix)):
#         y_axis = np.zeros((len(labels)))
#         values = np.zeros((len(labels)))
#         predicted_class_index = 0
#         for value in matrix[true_label_index]:
#             if value!=0:
#                 y_axis[predicted_class_index] = true_label_index
#                 values[predicted_class_index] = value
#             ax.scatter(x_axis, y_axis)
#             for i, txt in enumerate(values):
#                 ax.annotate(txt, (x_axis[i], y_axis[i]))
#             predicted_class_index += 1
#
#     plt.xlabel('Predicted Class')
#     plt.ylabel('True Class')
#     if is_training:
#         plt.title('Confusion Matrix for Training Data', fontsize=18)
#     else:
#         plt.title('Confusion Matrix for Test Data', fontsize=18)
#     plt.axis([0, len(labels), 0, len(labels)])
#     my_xticks = labels
#     plt.xticks(x_axis, my_xticks)
#     plt.yticks(x_axis, my_xticks)
#     for tick in ax.xaxis.get_major_ticks():
#             # tick.label.set_fontsize(2)
#             tick.label.set_rotation('vertical')
#     # for tick in ax.yaxis.get_major_ticks():
#     #         tick.label.set_fontsize(2)
#     plt.show()
#     return matrix


def create_heatmap(project, matrix, figure_path):
    """
    This function displyas the heatmap,
    according to the given matrix.
    """
    number_of_test_species = len(project.species) - 2
    # fig = plt.figure(figsize=(15, 7))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Format
    # fig = plt.gcf()
    # fig.set_size_inches(8, 11)
    # turn off the frame
    ax.set_frame_on(False)
    ax.grid(False)
    new_y_axis = [i for i in range(number_of_test_species)]
    plt.xlabel('Train', fontsize=18, fontweight='bold')
    ax.xaxis.set_label_coords(0.5, -0.07)
    plt.ylabel('Test', fontsize=18, fontweight='bold')
    ax.yaxis.set_label_coords(-0.15, 0.5)
    for train_index in range(len(project.species)):
        new_x_axis = np.zeros(len(project.species))
        values = np.zeros(len(project.species)-2)
        test_index = 0
        for value in matrix[train_index]:
            new_x_axis[test_index] = train_index
            values[test_index] = value
            # ax.scatter(new_y_axis, new_x_axis)
            # for i, txt in enumerate(values):
            #     ax.annotate(txt, (new_y_axis[i]+0.5, new_x_axis[i]+0.5),horizontalalignment='center',
            #                     verticalalignment='center', fontsize=20)
            test_index += 1
        for i, txt in enumerate(values):
            ax.annotate(txt, (new_x_axis[i] + 0.5, new_y_axis[i] + 0.5), horizontalalignment='center',
                    verticalalignment='center', fontsize=15)


    transposed_matrix = np.transpose(matrix)
    # heatmap = plt.pcolor(matrix)  # for all colors
    # transposed_heatmap = plt.pcolor(transposed_matrix)  # for all colors
    transposed_heatmap = plt.pcolor(transposed_matrix, cmap=plt.cm.autumn) # yellow-red colormap
    plt.clim(0.8, 1)
    # cbar = plt.colorbar(heatmap, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    cbar = plt.colorbar(transposed_heatmap, ticks=[0.8, 0.85, 0.9, 0.95, 1])
    # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])

    # cbar.ax.get_yaxis().set_ticks([])
    # for j, value in enumerate([float(i/10) for i in range(11)]):
    #     cbar.ax.text(value, ha='center', va='center', fontsize=20)
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", "3%", pad="1%")
    cbar.ax.tick_params(labelsize=10)

    # cbar.tick_params(labelsize=20)
    if project.project_name == "TF_vs_k_shuffle":
        plt.suptitle('TF vs. Shuffled TF with k='+str(project.k)+'\n'
                         'AUC Results for CNNs trained and tested on different species',
                         fontsize=14)
    elif project.project_name == "TF_vs_negative_data":
        plt.suptitle('TF vs. negative data\n'
                     'AUC Results for CNNs trained and tested on different species',
                     fontsize=14)
    species_names_map = {"Canis_familiaris": "Dog", "Homo_sapiens": "Human",
                         "Monodelphis_domestica": "Opossum", "Mus_musculus": "Mouse"}
    new_labels = []
    for species_latin_name in project.species[:number_of_test_species]:
        short_name = species_names_map[species_latin_name]
        new_labels.append(short_name)
    new_labels.append("All 12K")
    new_labels.append("All 60K")

    plt.xticks(new_x_axis, new_labels)
    my_xticks = new_labels
    plt.yticks(new_y_axis, new_labels)
    plt.xticks(new_x_axis, my_xticks)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            # tick.label.set_rotation(15)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    vector_yticks_locations = list(range(1, len(project.species)-1))
    vector_yticks_locations = [x - 0.5 for x in vector_yticks_locations]
    ax.set_yticks(vector_yticks_locations)

    vector_xticks_locations = list(range(1, len(project.species)+1))
    vector_xticks_locations = [x - 0.5 for x in vector_xticks_locations]
    ax.set_xticks(vector_xticks_locations)

    fig.savefig(figure_path, format='pdf')
    # plt.show()


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'display_heatmap_TF.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)

    if "Gallus_gallus" in project.species:
        indices_map = {"Canis_familiaris": 0, "Gallus_gallus": 1, "Homo_sapiens": 2,
                   "Monodelphis_domestica": 3, "Mus_musculus": 4,
                   "All_species_12000": 5, "All_species_60000": 6}
    else:
        indices_map = {"Canis_familiaris": 0,  "Homo_sapiens": 1,
                       "Monodelphis_domestica": 2, "Mus_musculus": 3,
                       "All_species_12000": 4, "All_species_60000": 5}


    test_results_file_1 = project.test_file
    # test_results_file_2 = os.path.join(project.CNN_output_dir, "CNN_test_output_k_4_train_on_all_species_60000_only.txt")
    number_of_test_species = len(project.species)-2
    trained_on_Cfam_results = [None] * number_of_test_species
    trained_on_Mmus_results = [None] * number_of_test_species
    if "Gallus_gallus" in project.species:
        trained_on_Ggal_results = [None] * number_of_test_species
    trained_on_Mdom_results = [None] * number_of_test_species
    trained_on_Hsap_results = [None] * number_of_test_species
    trained_on_All_12000_results = [None] * number_of_test_species
    trained_on_All_60000_results = [None] * number_of_test_species
    matrix_results_CEBPA_k_4 = []
    auc = None
    for test_results_file in [test_results_file_1]:
        with open(test_results_file) as results_file:
            for line in results_file:
                if re.match("^\s$", line):
                    continue
                elif line.startswith("train:") or \
                        line.startswith("finish test"):
                    continue
                split_line = line.split()
                # print("line = ",line)
                if line.startswith("best_model_validation_id"):
                    model_id = split_line[2]
                    train = map_model_ids[model_id]

                # if line.startswith("train:"):
                #     train = split_line[1]
                elif line.startswith("test:"):
                    test = split_line[1]
                    if "All_species" in test:
                        break
                elif line.startswith("auc:"):
                    auc = float(split_line[1])

                # res = Result(k_str, best_model_validation_id, train, test, test_accuracy)
                # print("test: ",test)
                # print("species_indices_map[test] = ", species_indices_map[test])
                if auc:
                    test_index = indices_map[test]
                    if train == "Canis_familiaris":
                        trained_on_Cfam_results[test_index] = auc
                    elif train == "Mus_musculus":
                        trained_on_Mmus_results[test_index] = auc
                    elif train == "Gallus_gallus":
                        trained_on_Ggal_results[test_index] = auc
                    elif train == "Monodelphis_domestica":
                        trained_on_Mdom_results[test_index] = auc
                    elif train == "Homo_sapiens":
                        trained_on_Hsap_results[test_index] = auc
                    elif train == "All_species_12000":
                        trained_on_All_12000_results[test_index] = auc
                    elif train == "All_species_60000":
                        trained_on_All_60000_results[test_index] = auc
                        # if None not in trained_on_All_60000_results:
                        #     break
    if project.k:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_TF_vs_k_shuffle_k_"+str(project.k)+".pdf")
    else:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_TF_vs_negative_data.pdf")
        results_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_results_TF_vs_negative_data.txt")
    with open(results_path, "w") as out_results:
        matrix_results_CEBPA_k_4.append(trained_on_Cfam_results)
        # print("trained_on_Cfam_results = ",trained_on_Cfam_results)
        string_list = "Dog: "
        for i in trained_on_Cfam_results:
            string_list += str(i)+","
        out_results.write(string_list+"\n")
        if "Gallus_gallus" in project.species:
            matrix_results_CEBPA_k_4.append(trained_on_Ggal_results)
            string_list = "Chicken: "
            for i in trained_on_Ggal_results:
                string_list += str(i) + ","
            out_results.write(string_list + "\n")
        matrix_results_CEBPA_k_4.append(trained_on_Hsap_results)
        string_list = "Human: "
        for i in trained_on_Hsap_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")
        matrix_results_CEBPA_k_4.append(trained_on_Mdom_results)
        string_list = "Opossum: "
        for i in trained_on_Mdom_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")
        matrix_results_CEBPA_k_4.append(trained_on_Mmus_results)
        string_list = "Mouse: "
        for i in trained_on_Mmus_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")
        matrix_results_CEBPA_k_4.append(trained_on_All_12000_results)
        string_list = "All_12000: "
        for i in trained_on_All_12000_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")
        matrix_results_CEBPA_k_4.append(trained_on_All_60000_results)
        string_list = "All_60000: "
        for i in trained_on_All_60000_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")

    create_heatmap(project, matrix_results_CEBPA_k_4, figure_path)
    print("End!!!")

if __name__ == "__main__":
    main()
