__author__ = 'Dikla Cohn'
"""
display_heatmap_enhancers.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TThis module is responsible for displaying the test AUC results (as a heatmap) 
of networks trained and tested on different species.
"""
__author__ = 'Dikla Cohn'
import numpy as np
import sys
import os
import test_CNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


titles = False

old_species_name_order = ["Human",
                          "Dog",
                          "Dolphin",
                          "Tasmanian_Devil",
                          "Ferret",
                          "Guinea_pig",
                          "Tree_shrew",
                          "Marmoset",
                          "Cat",
                          "Cow",
                          "Opossum",
                          "Mouse",
                          "Macaque",
                          "Rabbit",
                          "Naked_mole_rat",
                          "Rat",
                          "Pig"]


def create_heatmap(project, matrix, figure_path):
    """
    This function displays the heatmap,
    according to the given matrix.
    """

    number_of_test_species = len(project.species) - 2
    new_matrix = np.zeros((number_of_test_species, number_of_test_species+2))
    for test_index in range(number_of_test_species):
        for i in range(len(matrix[test_index])):
            value = matrix[test_index][i]
            new_matrix[number_of_test_species-1-test_index, i] = value

    matrix = new_matrix

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # turn off the frame
    ax.set_frame_on(False)
    ax.grid(False)
    new_y_axis = [i for i in range(number_of_test_species)]
    plt.xlabel('Train', fontsize=18, fontweight='bold')
    ax.xaxis.set_label_coords(0.64, -0.15)
    plt.ylabel('Test', fontsize=18, fontweight='bold')
    ax.yaxis.set_label_coords(-0.14, 0.5)

    for test_index in range(number_of_test_species):
        new_x_axis = np.zeros(len(project.species))
        values = np.zeros(len(project.species))
        train_index = 0
        for value in matrix[test_index]:
            new_x_axis[train_index] = test_index
            values[train_index] = value
            ax.annotate('{0:0.2f}'.format(value),
                        (train_index + 0.5, test_index + 0.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8.3)
            train_index += 1

    heatmap = plt.pcolor(matrix, cmap=plt.cm.YlOrRd)  # yellow-red colormap)
    plt.clim(0.5, 1)

    if titles:
        if project.project_name == "H3K27ac_vs_k_shuffle":
            plt.suptitle('H3K27ac vs. Shuffled TF with k='+str(project.k)+'\n'
                             'AUC Results for CNNs trained and tested on different species',
                             fontsize=14)
        elif project.project_name == "H3K27ac_vs_negative_data":
            plt.suptitle('H3K27ac vs. negative data\n'
                         'AUC Results for CNNs trained and tested on different species',
                         fontsize=14)
    new_labels_y = []
    new_labels_x = []
    for species_name in project.species[:number_of_test_species]:
        if species_name == "Tasmanian_Devil":
            new_label_x = "Tasmanian\n devil"
            new_label_y = "Tasmanian\ndevil  "
        elif species_name == "Naked_mole_rat":
            new_label_x = "Naked\n mole-rat"
            new_label_y = "Naked \nmole-rat"
        elif species_name == "Guinea_pig":
            new_label_x = "Guinea pig"
            new_label_y = new_label_x
        elif species_name == "Tree_shrew":
            new_label_x = "Tree shrew"
            new_label_y = new_label_x
        else:
            new_label_x = species_name
            new_label_y = new_label_x
        new_labels_x.append(new_label_x)
        new_labels_y.append(new_label_y)

    new_labels_y.reverse()
    new_labels_x.append("All 14K")
    new_labels_x.append("All 238K")
    plt.xticks(new_x_axis, new_labels_x)
    plt.yticks(new_y_axis, new_labels_y)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
            tick.label.set_rotation(90)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
    vector_xticks_locations = list(range(1, len(project.species) + 1))
    vector_xticks_locations = [x - 0.5 for x in vector_xticks_locations]
    ax.set_xticks(vector_xticks_locations)
    vector_yticks_locations = list(range(1, number_of_test_species + 1))
    vector_yticks_locations = [x - 0.5 for x in vector_yticks_locations]
    ax.set_yticks(vector_yticks_locations)
    plt.subplots_adjust(top=0.99, bottom=-0.08, left=0.16, right=0.98)
    cbar = plt.colorbar(heatmap, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        orientation='horizontal', aspect=40)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(figure_path, format='pdf')



def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'display_heatmap_enhancers.py')
    test_results_file_path = os.path.join(project.CNN_output_dir, "test_results.txt")
    number_of_test_species = len(project.species) - 2
    tested_on_all_species = []
    for i in range(number_of_test_species):
        tested_on_one_species = [None] * len(project.species)
        tested_on_all_species.append(tested_on_one_species)
    test_counter = 0
    number_of_values = 0
    sum_v = 0
    with open(test_results_file_path) as results_file:
        for line in results_file:
            if "\n" in line:
                new_line = line[:-1]
            if new_line in project.species:
                train_index = project.species.index(new_line)
                continue
            split_line = new_line.split()
            for value in split_line:
                number_of_values += 1
                sum_v += float(value)
                old_test_species = old_species_name_order[test_counter]
                new_test_index = project.species.index(old_test_species)
                tested_on_all_species[new_test_index][train_index] = float(value)
                test_counter += 1
                if test_counter == number_of_test_species:
                    test_counter = 0

    figure_path = os.path.join(project.CNN_output_dir,
                               "heatmap_enhancers_vs_negative_data_cbar.pdf")

    create_heatmap(project, tested_on_all_species, figure_path)

if __name__ == "__main__":
    main()
