import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
from DataLoader import DataLoader
sys.path.insert(0, base_path[:-len('/simulated_data')]+'/CNN/')
import test_CNN


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, "run_data_loader.py")
    print "start creating data for project: ", project.project_name
    data_loader = DataLoader(project)
    data_loader.create_npy_files()
    print "End!"

if __name__ == "__main__":
    main()
