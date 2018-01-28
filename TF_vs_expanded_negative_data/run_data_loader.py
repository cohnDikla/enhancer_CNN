"""
Creates the TF data vs. the expanded negative data (non-enhancers + k-shuffle), for all species.
Sets a fixed partition of train, validation and test data for each species.
write all created data samples and labels both as text files and as numpy binary files.
run with python2.7 !!!
"""
import os
import sys
from DataLoaderTFvsExpandedNeg import DataLoaderTFvsExpandedNeg
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
projects_base_path = base_path[:-len('/TF_vs_expanded_negative_data')]
sys.path.insert(0, projects_base_path+'/CNN/')
import test_CNN


def main():
    num_times_negative_data_is_taken = 2
    project = test_CNN.get_project_and_check_arguments(sys.argv, "run_data_loader.py",
                                                       num_times_negative_data_is_taken=num_times_negative_data_is_taken)
    print "start creating data for project: ", project.project_name
    for species in project.species:
        species_text_samples_dir = os.path.join(project.text_samples_base_dir, species)
        species_npy_samples_dir = os.path.join(project.samples_base_dir, species)
        if not os.path.exists(species_text_samples_dir) and not os.path.isdir(species_text_samples_dir):
            print("make directory: ", species_text_samples_dir)
            os.makedirs(species_text_samples_dir)
        if not os.path.exists(species_npy_samples_dir) and not os.path.isdir(species_npy_samples_dir):
            print("make directory: ", species_npy_samples_dir)
            os.makedirs(species_npy_samples_dir)
    data_loader = DataLoaderTFvsExpandedNeg(project)
    data_loader.get_all_positive_and_negative_samples(num_times_negative_data_is_taken)

    data_loader.create_data_for_each_species()
    # data_loader.create_data_from_all_species_together()
    print "End!"

if __name__ == "__main__":
    main()
