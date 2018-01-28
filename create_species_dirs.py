"""
create a directory for each species in the 'data' dir ('npy_files' and 'samples' dirs)
of the given project.
usage:
python3 /create_species_dirs.py <project_name>
"""

import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path+'/CNN/')
import test_CNN



def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'create_species_dirs.py')
    project_data_dir = os.path.join(base_path, project.project_name, 'data')
    for dir_name in ['npy_files', 'samples']:
        dir_path = os.path.join(project_data_dir, dir_name)
        for species in project.species:
            species_dir = os.path.join(dir_path, species)
            if not os.path.exists(species_dir) and not os.path.isdir(species_dir):
                print("make directory: ", species_dir)
                os.makedirs(species_dir)

if __name__ == "__main__":
    main()