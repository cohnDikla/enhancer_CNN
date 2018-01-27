__author__ = 'Dikla Cohn'
"""
display_heatmap_H3K27ac.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module is responsible for displaying a heatmap of all test AUC results,
trained and tested on each species separately.
"""
import os
import numpy as np
import test_CNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys


def create_heatmap(project, matrix, figure_path):
    """
    This function displyas the heatmap,
    according to the given matrix.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    ax.grid(False)
    number_of_test_species = len(project.species)-2
    x_axis = [i for i in range(number_of_test_species)]
    plt.xlabel('Test', fontsize=10, fontweight='bold')
    ax.xaxis.set_label_coords(0.5, -0.07)
    plt.ylabel('Train', fontsize=10, fontweight='bold')
    ax.yaxis.set_label_coords(-0.15, 0.5)
    for train_index in range(len(project.species)):
        y_axis = np.zeros(len(project.species))
        values = np.zeros(len(project.species)-2)
        test_index = 0
        for value in matrix[train_index]:
            y_axis[test_index] = train_index
            values[test_index] = value
            test_index += 1
        for i, txt in enumerate(values):
            ax.annotate(txt, (x_axis[i] + 0.5, y_axis[i] + 0.5), horizontalalignment='center',
                    verticalalignment='center', fontsize=5)

    heatmap = plt.pcolor(matrix)  # TODO for all colors, not only Blues
    plt.clim(0.5, 1)
    cbar = plt.colorbar(heatmap, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    cbar.ax.tick_params(labelsize=10)
    if project.project_name == "H3K27ac_vs_k_shuffle":
        plt.suptitle('H3K27ac vs. Shuffled H3K27ac with k='+str(project.k)+'\n'
                         'AUC Results for CNNs trained and tested on different species',
                         fontsize=14)
    elif project.project_name == "H3K27ac_vs_negative_data":
        plt.suptitle('H3K27ac vs. negative data\n'
                     'AUC Results for CNNs trained and tested on different species',
                     fontsize=14)

    plt.axis([0, number_of_test_species, 0, len(project.species)])
    new_labels = project.species[:number_of_test_species]
    new_labels.append("All species 14K")
    new_labels.append("All species 238K")
    new_labels[project.species.index("Tasmanian_Devil")] = "Tasmanian Devil"
    new_labels[project.species.index("Tree_shrew")] = "Tree shrew"
    new_labels[project.species.index("Naked_mole_rat")] = "Naked mole rat"
    new_labels[project.species.index("Guinea_pig")] = "Guinea pig"
    plt.yticks(y_axis, new_labels)

    new_labels_x = new_labels
    new_labels_x[project.species.index("Tasmanian_Devil")] = "Tasmanian"+"\n"+"Devil"
    new_labels[project.species.index("Naked_mole_rat")] = "Naked\nmole rat"
    plt.xticks(x_axis, new_labels)

    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
            tick.label.set_rotation(55)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)

    vector_xticks_locations = list(range(1, len(project.species)-1))
    vector_xticks_locations = [x - 0.5 for x in vector_xticks_locations]
    idx = project.species.index("Tasmanian_Devil")
    vector_xticks_locations[idx] = (vector_xticks_locations[idx]-0.25)
    idx = project.species.index("Naked_mole_rat")
    vector_xticks_locations[idx] = (vector_xticks_locations[idx] - 0.25)

    ax.set_xticks(vector_xticks_locations)
    vector_yticks_locations = list(range(1, len(project.species)+1))
    vector_yticks_locations = [x - 0.5 for x in vector_yticks_locations]
    ax.set_yticks(vector_yticks_locations)
    fig.savefig(figure_path, format='pdf')

def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'display_heatmap_H3K27ac.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)
    test_results_file = project.test_file
    number_of_test_species = len(project.species) - 2
    indices_map = dict()
    i = 0
    for species in project.species:
        indices_map[species] = i
        i += 1

    trained_on_all_species_results = []
    for i in range(len(project.species)):
        trained_on_one_species_results = [None] * number_of_test_species
        trained_on_all_species_results.append(trained_on_one_species_results)

    matrix_results_CEBPA_one_k = []

    auc = None
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
                train_index = indices_map[train]
            elif line.startswith("test:"):
                test = split_line[1]
                if "All_species" in test:
                    break
            elif line.startswith("auc:"):
                auc = float(split_line[1])
            if auc:
                test_index = indices_map[test]

                trained_on_all_species_results[train_index][test_index] = float(auc)

    print("len(trained_on_all_species_results) = ", len(trained_on_all_species_results))

    print("number_of_test_species = ", number_of_test_species)
    print("(((number_of_test_species)*(number_of_test_species)) +len(project.species)) = ",
          (((number_of_test_species) * (number_of_test_species)) + len(project.species)))

    average_train_all_species = sum(trained_on_all_species_results[-1]) / number_of_test_species
    print("average_train_all_species = ", average_train_all_species)
    for i in range(number_of_test_species):
        matrix_results_CEBPA_one_k.append(trained_on_all_species_results[i])
    matrix_results_CEBPA_one_k.append(trained_on_all_species_results[number_of_test_species+1])
    matrix_results_CEBPA_one_k.append(trained_on_all_species_results[number_of_test_species])

    if project.k:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_test_CNN_k_"+str(project.k)+".pdf")
    else:
        figure_path = os.path.join(project.CNN_output_dir,
                                   "heatmap_test_CNN.pdf")
    create_heatmap(project, matrix_results_CEBPA_one_k, figure_path)

    print("End!!!")

if __name__ == "__main__":
    main()
