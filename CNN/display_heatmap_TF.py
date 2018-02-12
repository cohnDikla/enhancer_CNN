__author__ = 'Dikla Cohn'
"""
display_heatmap_TF.py
~~~~~~~~~~~~~~~~~~~~~~
This module is responsible for displaying the test AUC results (as a heatmap) 
of networks trained and tested on different species.
"""
__author__ = 'Dikla Cohn'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys
import os
import numpy as np
import test_CNN

titles = False


def create_heatmap(project, matrix, figure_path):
    """
    This function displays the heatmap,
    according to the given matrix.
    """
    number_of_test_species = len(project.species) - 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # turn off the frame
    ax.set_frame_on(False)
    ax.grid(False)
    new_y_axis = [i for i in range(number_of_test_species)]
    plt.xlabel('Train', fontsize=18, fontweight='bold')
    # ax.xaxis.set_label_coords(0.5, -0.07)
    plt.ylabel('Test', fontsize=18, fontweight='bold')
    # ax.yaxis.set_label_coords(-0.15, 0.5)
    for test_index in range(number_of_test_species):
        new_x_axis = np.zeros(len(project.species))
        values = np.zeros(len(project.species))
        train_index = 0
        for value in matrix[test_index]:
            new_x_axis[train_index] = test_index
            values[train_index] = value
            ax.annotate('{0:0.3f}'.format(value),
                        (train_index + 0.5, test_index + 0.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=13)
            train_index += 1

    heatmap = plt.pcolor(matrix, cmap=plt.cm.YlOrRd)  # yellow-red colormap
    plt.clim(0.8, 1)
    cbar = plt.colorbar(heatmap, ticks=[0.8, 0.85, 0.9, 0.95, 1])
    cbar.ax.tick_params(labelsize=10)

    if titles:
        if project.project_name == "TF_vs_k_shuffle":
            plt.suptitle('TF vs. Shuffled TF with k='+str(project.k)+'\n'
                             'AUC Results for CNNs trained and tested on different species',
                             fontsize=14)
        elif project.project_name == "TF_vs_negative_data":
            plt.suptitle('TF vs. negative data\n'
                         'AUC Results for CNNs trained and tested on different species',
                         fontsize=14)

    new_labels_x = ["Human", "Mouse", "Dog", "Opossum", "All 12K", "All 48K"]
    plt.xticks(new_x_axis, new_labels_x)
    new_labels_y = ["Opossum", "Dog", "Mouse", "Human"]
    plt.yticks(new_y_axis, new_labels_y)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
            tick.label.set_rotation(19)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    vector_yticks_locations = list(range(1, len(project.species)-1))
    vector_yticks_locations = [x - 0.5 for x in vector_yticks_locations]
    ax.set_yticks(vector_yticks_locations)

    vector_xticks_locations = list(range(1, len(project.species)+1))
    vector_xticks_locations = [x - 0.5 for x in vector_xticks_locations]
    ax.set_xticks(vector_xticks_locations)

    plt.subplots_adjust(top=0.98, bottom=0.18, left=0.22, right=1)
                        # hspace=0.25,
                        # wspace=0.35)
    fig.savefig(figure_path, format='pdf')

def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'display_heatmap_TF.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)

    if "Gallus_gallus" in project.species:
        train_indices_map = {"Canis_familiaris": 2, "Gallus_gallus": 4, "Homo_sapiens": 0,
                   "Monodelphis_domestica": 3, "Mus_musculus": 1,
                   "All_species_12000": 5, "All_species_60000": 6}
    else:
        train_indices_map = {"Canis_familiaris": 2,  "Homo_sapiens": 0,
                       "Monodelphis_domestica": 3, "Mus_musculus": 1,
                       "All_species_12000": 4, "All_species_60000": 5}

    test_results_file_1 = project.test_file
    tested_on_Cfam_results = [None] * len(project.species)
    tested_on_Mmus_results = [None] * len(project.species)
    if "Gallus_gallus" in project.species:
        tested_on_Ggal_results = [None] * len(project.species)
    tested_on_Mdom_results = [None] * len(project.species)
    tested_on_Hsap_results = [None] * len(project.species)

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
                if line.startswith("best_model_validation_id"):
                    model_id = split_line[2]
                    train = map_model_ids[model_id]
                elif line.startswith("test:"):
                    test = split_line[1]

                elif line.startswith("auc:"):
                    auc = float(split_line[1])
                if auc:
                    train_index = train_indices_map[train]
                    if test == "Canis_familiaris":
                        tested_on_Cfam_results[train_index] = auc
                    elif test == "Mus_musculus":
                        tested_on_Mmus_results[train_index] = auc
                    elif test == "Gallus_gallus":
                        tested_on_Ggal_results[train_index] = auc
                    elif test == "Monodelphis_domestica":
                        tested_on_Mdom_results[train_index] = auc
                    elif test == "Homo_sapiens":
                        tested_on_Hsap_results[train_index] = auc

    if project.k:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_TF_vs_k_shuffle_k_"+str(project.k)+".pdf")
    else:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_TF_vs_negative_data.pdf")
        results_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_results_TF_vs_negative_data.txt")
    with open(results_path, "w") as out_results:
        if "Gallus_gallus" in project.species:
            matrix_results_CEBPA_k_4.append(tested_on_Ggal_results)
            string_list = "Chicken: "
            for i in tested_on_Ggal_results:
                string_list += str(i) + ","
            out_results.write(string_list + "\n")

        matrix_results_CEBPA_k_4.append(tested_on_Mdom_results)
        string_list = "Opossum: "
        for i in tested_on_Mdom_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")

        matrix_results_CEBPA_k_4.append(tested_on_Cfam_results)
        string_list = "Dog: "
        for i in tested_on_Cfam_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")

        matrix_results_CEBPA_k_4.append(tested_on_Mmus_results)
        string_list = "Mouse: "
        for i in tested_on_Mmus_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")

        matrix_results_CEBPA_k_4.append(tested_on_Hsap_results)
        string_list = "Human: "
        for i in tested_on_Hsap_results:
            string_list += str(i) + ","
        out_results.write(string_list + "\n")

    create_heatmap(project, matrix_results_CEBPA_k_4, figure_path)

if __name__ == "__main__":
    main()
