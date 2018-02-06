"""
Creates the H3K27ac data vs. the new negative non-enhancers data, for all species.
Sets a fixed partition to train, validation and test data for each species.
write all created data samples and labels both as text files and as numpy binary files.
run with python2.7 !!!
"""
import os
import sys
from DataLoaderH3K27acvsShuffle import DataLoaderH3K27acvsShuffle
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
projects_base_path = base_path[:-len('/H3K27ac_vs_k_shuffle')]
sys.path.insert(0, projects_base_path+'/CNN/')
import test_CNN


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, "run_data_loader.py")
    print "start creating data for project: ", project.project_name
    for species in project.species:
        species_text_samples_dir = os.path.join(project.text_samples_base_dir, species,
                                                project.k_let_dirs[project.k-1])
        species_npy_samples_dir = os.path.join(project.samples_base_dir, species)
        if not os.path.exists(species_text_samples_dir) and not os.path.isdir(species_text_samples_dir):
            print("make directory: ", species_text_samples_dir)
            os.makedirs(species_text_samples_dir)
        if not os.path.exists(species_npy_samples_dir) and not os.path.isdir(species_npy_samples_dir):
            print("make directory: ", species_npy_samples_dir)
            os.makedirs(species_npy_samples_dir)

    data_loader = DataLoaderH3K27acvsShuffle(project)
    data_loader.get_all_positive_and_negative_samples()
    
    data_loader.create_data_from_all_species_together()
    print "End!"

if __name__ == "__main__":
    main()
