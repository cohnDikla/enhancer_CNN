import sys
import os
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import test_CNN
import tarfile

plot_ROC = False
trained_on_one_species_only = True
train_on_all_samples = False
train_on_human = True
train_on_dog = False

# n = 2  # TODO update
n = None

def main():
    # os.system("module load tensorflow;")
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'run_test_CNN.py')
    project.num_times_negative_data_is_taken = n
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)
    print("map_model_ids: ", map_model_ids)
    # output_dir = os.path.join(project.CNN_output_dir,
    #                           project.distribution_samples_center_dir,
    #                           project.PWM)
    sum_auc = 0
    with open(project.test_file, 'a+') as out_file:
        number_of_species = len(project.species)
        number_species_tested_on = 0
        for index_to_test_on in range(number_of_species + 1):
            if index_to_test_on != number_of_species:
                test_species = project.species[index_to_test_on]
                if "All_species" in test_species:
                    continue
                number_species_tested_on += 1
                print("test on species: ", test_species)
            for best_model_validation_id in sorted_models_list:
                # last_iteration - just plot the average auc of all tested species on the ROC figure
                if index_to_test_on == number_of_species:
                    average_auc_returned = test_CNN.draw_roc_curve(array_true_labels, array_prediction_scores, project, model_dir,
                                                  best_model_validation_id, out_file, train_species,
                                                  test_species, plot_ROC=plot_ROC, average_auc=average_auc)
                    print("average_auc_returned : ", average_auc_returned)
                    out_file.write("average_auc_returned: {0:.3f}".format(average_auc_returned) + "\n")
                    break
                train_species = map_model_ids[best_model_validation_id]
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
                    train_species = project.species[trained_species_index]
                    print("train_species: ", train_species)
                # if project.project_name == "TF_vs_k_shuffle":
                #     model_dir = os.path.join(project.checkpoints_folder, best_model_validation_id)
                # else:
                model_dir = os.path.join(project.checkpoints_folder_tmp, best_model_validation_id)
                if not os.path.exists(model_dir) and not os.path.isdir(model_dir):
                    tar = tarfile.open(model_dir + ".tar")
                    tar.extractall(path=model_dir)
                    tar.close()
                print("train on species: ", train_species)
                model_dir = test_CNN.create_directories(project, best_model_validation_id)
                array_true_labels, array_prediction_scores = \
                    test_CNN.import_model_and_test(project, best_model_validation_id,
                                                   test_species, train_species, out_file)
                test_CNN.write_labels_and_scores(model_dir, best_model_validation_id,
                                                 array_prediction_scores, array_true_labels, test_species)
                auc = test_CNN.draw_roc_curve(array_true_labels, array_prediction_scores, project,
                                              model_dir, best_model_validation_id, out_file,
                                              train_species, test_species, plot_ROC=plot_ROC)
                print("auc: ", auc)
                # print("sum_auc: ", sum_auc)
                sum_auc += auc
                average_auc = (sum_auc / number_species_tested_on)
            if index_to_test_on != number_of_species:
                out_file.write('finish test on species: '+test_species+'\n\n\n')
                out_file.flush()
            print()
        # print("average_auc : ", average_auc)
        # out_file.write("average AUC: {0:.3f}".format(average_auc) + "\n")

    print("end")


if __name__ == "__main__":
    main()
