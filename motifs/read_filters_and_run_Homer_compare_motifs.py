"""
For each species, creates motifs file from the learned filters, for running Homer's script:
compareMotifs.pl:
Usage: compareMotifs.pl <motifs file> <output directory> [options]
and runs the compareMotifs script with this file and vertebrates known motifs.
For example:
~tommy/Work/HOMER/bin/compareMotifs.pl /cs/grad/diklac03/Work/projects/Five-vertebrate_ChIP-seq/output/CNN_2_conv_layers_filters/preserving_4-let_counts/Homo_sapiens/filters_file.txt /cs/grad/diklac03/Work/projects/Five-vertebrate_ChIP-seq/output/CNN_2_conv_layers_filters/preserving_4-let_counts/Homo_sapiens/Homer_compareMotifs/ -known /cs/cbio/tommy/HOMER/data/knownTFs/vertebrates/all.motifs
"""
import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path[:-len('/motifs')]+'/CNN/')
import data_handle
import test_CNN

trained_on_all_species_only = False
# trained_species_index = 0

# filters_folder = "/cs/cbio/dikla/projects/20-Enh/output/CNN_version_2_filters_all_k/" \
#                       "filters_conv_1/preserving_4-let_counts/"
# filters_folder = "/cs/cbio/dikla/projects/" \
#                      "Five-vertebrate_ChIP-seq/output/CNN_2_conv_layers_filters/" \
#                      "preserving_4-let_counts/"


def create_filters_file_for_Homer_compareMotifs(filters_folder, num_filters_in_first_layer,
                                                species_name):
    """
    The filter file should be a text file where each filter start with s header line of the form:
    >filter_name
    and then 4 columns of numbers, for A,C,G,T.
    number of row is equal to the motif/filter length.
    :return:
    """
    output_file_path = os.path.join(filters_folder, "filters_file.txt")
    with open(output_file_path, "w") as out:
        for i in range(1, num_filters_in_first_layer + 1):
            filter_file_path = os.path.join(filters_folder,
                                            "filter" + str(i) + "_" + species_name + ".txt")
            array_counts_all_bases = data_handle.read_PWM_from_file(filter_file_path)
            # print("np.size(array_values_all_positions) = ", np.size(array_values_all_positions))
            # print("len(array_values_all_positions) = ", len(array_values_all_positions))
            # print("len(array_counts_all_bases[0]) = ", len(array_counts_all_bases[0]))
            values_all_bases = []
            for base_index in range(len(array_counts_all_bases)):
                values_one_base = []
                for position_index in range(len(array_counts_all_bases[0])):
                    value = array_counts_all_bases[base_index, position_index]
                    values_one_base.append(value)
                values_all_bases.append(values_one_base)

            out.write(">filter" + str(i) + "\n")
            for position_index in range(len(array_counts_all_bases[0])):
                for base_index in range(len(array_counts_all_bases)):
                    value = values_all_bases[base_index][position_index]
                    out.write(str(value) + "\t")
                out.write("\n")


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv,
                                                       'read_filters_and_run_Homer_compare_motifs.py')
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)

    for best_model_validation_id in sorted_models_list:
        train_species = map_model_ids[best_model_validation_id]
        if trained_on_all_species_only:
            trained_species_index = len(project.species) - 2  # TODO update
            if train_species != project.species[trained_species_index]:
                continue
        print("train_name = ", train_species)

        model_dir = test_CNN.create_directories(project, best_model_validation_id)
        conv_results_dir = os.path.join(model_dir, 'convolution_results')
        filters_folder = os.path.join(conv_results_dir, "filters",  "layer_1")
        num_filters_in_first_layer = project.CNN_structure.get_kernels_shape_and_number(1)[1]
        create_filters_file_for_Homer_compareMotifs(filters_folder, num_filters_in_first_layer,
                                                    train_species)
        # run Homer:
        script = "~tommy/Work/HOMER/bin/compareMotifs.pl"
        motifs_file = os.path.join(filters_folder, "filters_file.txt")
        output_directory = os.path.join(filters_folder, "Homer_compareMotifs/")
        known_motifs = "/cs/cbio/tommy/HOMER/data/knownTFs/vertebrates/all.motifs"
        homer_results_file = os.path.join(filters_folder, "homer_results.txt")
        os.system(script + " " + motifs_file + " " + output_directory + " -known " + known_motifs +
                  " -cpu 2 2>&1 | tee " + homer_results_file)


    print("End!")


if __name__ == "__main__":
    main()


