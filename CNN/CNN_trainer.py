"""
CNN trainer module.
Trains the CNN according to the given project_name, num_runs and num_epochs.
Loads pickled object arrays of samples previously stored in npy files.
"""
__author__ = 'Dikla Cohn'
import os
import sys
import time
from CNN import CNN
from Project import Project
base_path = os.path.dirname(os.path.abspath(__file__))

trained_on_one_species_only = False
train_on_all_samples = False
train_on_human = False
train_on_dog = False

init_according_to_given_filters = False  # update for planted filters
init_model_ids = ["All_species_238000_k_2_20180116154913.UOUWIMmM",
                  "All_species_238000_k_3_20180116152307.qznh2WrM",
                  "All_species_238000_k_4_20180106224244.w9xbYWXj"]
n = None  # update


def get_project_and_check_arguments(argv, script_name):
    if len(argv) not in [4, 5]:
        sys.exit("Usage: "+script_name+" <project_name_and_PWM> <num_runs> <num_epochs> "
                 "[<k=None> or <normal_sigma>] \n")
    project_name = argv[1]
    num_runs = int(argv[2])
    num_epochs = int(argv[3])
    k = None
    is_normal_distribution = False
    sigma = None
    if len(argv) == 5:
        if argv[4].isdigit():
            k = int(argv[4])
        else:
            is_normal_distribution = bool(argv[4])  # True
            sigma = int(argv[4].split("_")[1])
    base_path_projects = base_path[:-len("CNN/")]
    if k:
        project = Project(project_name, base_path_projects, k=k)
    else:
        project = Project(project_name, base_path_projects,
                          normal_distribution=is_normal_distribution,
                          sigma=sigma)
    return project, num_runs, num_epochs



def create_directories(project):
    checkpoints_dir = os.path.join(project.project_base_path, "checkpoints")
    if not os.path.exists(checkpoints_dir) and not os.path.isdir(checkpoints_dir):
        print("make directory: ", checkpoints_dir)
        os.makedirs(checkpoints_dir)
    for k_let_dir in project.k_let_dirs:
        dir_k = os.path.join(checkpoints_dir, k_let_dir)
        if not os.path.exists(dir_k) and not os.path.isdir(dir_k):
            print("make directory: ", dir_k)
            os.makedirs(dir_k)


def create_directories_simulated_data(project):
    output_dir_dist = os.path.join(project.CNN_output_dir,
                                   project.distribution_samples_center_dir)
    if not os.path.exists(output_dir_dist) and not os.path.isdir(output_dir_dist):
        print("make directory: ", output_dir_dist)
        os.makedirs(output_dir_dist)
    output_dir = os.path.join(project.CNN_output_dir,
                              project.distribution_samples_center_dir,
                              project.PWM)
    if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
        print("make directory: ", output_dir)
        os.makedirs(output_dir)


def main():
    t = time.time()
    project, num_runs, num_epochs = get_project_and_check_arguments(sys.argv, "CNN_trainer.py")
    number_of_species = len(project.species)
    if project.k:
        create_directories(project)
    elif project.project_name == "simulated_data":
        create_directories_simulated_data(project)

    for i in range(number_of_species):
        if trained_on_one_species_only:
            if train_on_human:
                if "Homo_sapiens" in project.species:
                    trained_species_index = project.species.index("Homo_sapiens")
                elif "Human" in project.species:
                    trained_species_index = project.species.index("Human")
            elif train_on_all_samples:
                trained_species_index = number_of_species - 2  # all species 238000
            elif train_on_dog:
                if "Canis_familiaris" in project.species:
                    trained_species_index = project.species.index("Canis_familiaris")
                elif "Dog" in project.species:
                    trained_species_index = project.species.index("Dog")
            if i != trained_species_index:
                continue
        if project.project_name == "simulated_data":
            species_to_train_on = None
            str_species_to_train_on = "Simulated"
        else:
            species_to_train_on = i
            str_species_to_train_on = project.species[species_to_train_on]
        # create and evaluate the model
        net = CNN(project, num_epochs, num_runs, species_to_train_on=species_to_train_on,
                  k=project.k, n=n, init_according_to_given_filters=init_according_to_given_filters,
                  init_model_ids=init_model_ids, start_time=t)
        # evaluate without test:
        max_validation_accuracy, best_model_validation_id, best_run_validation_index = \
            net.evaluate()
        output_dir = project.CNN_output_dir
        if not os.path.exists(output_dir) and not os.path.isdir(output_dir):
            print("make directory: ", output_dir)
            os.makedirs(output_dir)
        # output the accuracy and params
        if not os.path.exists(project.output_results_file):
            open(project.output_results_file, 'w')
        with open(project.output_results_file, 'a') as f:  # appends the line at the end of the file
            f.write('train: {0}\t'
                    'max_validation_accuracy: {1:.3f}\t'
                    'best_model_validation_id: {2}\t'
                    'best_run_validation_index: {3}\t'
                    'k: {4}\t'
                    'num_epochs: {5}\t'
                    'num_runs: {6}\t'.format(str_species_to_train_on, max_validation_accuracy,
                                             best_model_validation_id, str(best_run_validation_index),
                                             str(project.k), str(num_epochs), str(num_runs)))
            f.flush()
        project.print_project_details(project.output_results_file)
    elapsed = time.time() - t
    print('Total training time (all runs): {0:0.2f} seconds'.format(elapsed))
    print("end of script! :)")

if __name__ == "__main__":
    main()
