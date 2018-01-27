import tensorflow as tf
import numpy as np
import os
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import sys
from Project import Project
from DataSetObject import DataSetObject
from SampleObject import SampleObject
from stat import S_ISREG, ST_CTIME, ST_MODE
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
# import tarfile
from matplotlib.backends.backend_pdf import PdfPages # for saving multiple plots to one pdf file
from matplotlib.font_manager import FontProperties
import re


def create_directories(project, model_id):
    if project.sigma:
        sigma_dir = os.path.join(project.base_dir_test_CNN_results, "sigma_"+str(project.sigma))
        if not os.path.exists(sigma_dir) and not os.path.isdir(sigma_dir):
            print("make directory: ", sigma_dir)
            os.makedirs(sigma_dir)

        model_dir = os.path.join(sigma_dir, model_id)
    else:
        model_dir = os.path.join(project.base_dir_test_CNN_results, model_id)
    if not os.path.exists(model_dir) and not os.path.isdir(model_dir):
        # print("is dir sample1? ", os.path.isdir(sample_dir))
        print("make directory: ", model_dir)
        os.makedirs(model_dir)
        convolution_results_dir = os.path.join(model_dir, 'convolution_results')
        print("make directory: ", convolution_results_dir)
        os.makedirs(convolution_results_dir)
    return model_dir


def get_project_and_check_arguments(argv, script_name, num_times_negative_data_is_taken=None):
    if len(argv) not in [2, 3]:
        sys.exit("Usage: python3 "+script_name+" <project_name_PWM_name> [<k=None> or <normal_sigma] \n")
    k = None
    sigma = None
    is_normal_distribution = False
    project_name = argv[1]
    if len(argv) == 3:
        if argv[2].isdigit():
            k = int(argv[2])
            print("k = ", k)
        else:
            is_normal_distribution = bool(argv[2])  # this is True
            sigma = int(argv[2].split("_")[1])
    base_path_projects = base_path[:-len("CNN/")]
    if k:
        project = Project(project_name, base_path_projects, k=k)
    else:
        project = Project(project_name, base_path_projects,
                          normal_distribution=is_normal_distribution, sigma=sigma,
                          num_times_negative_data_is_taken=num_times_negative_data_is_taken)
    return project


def get_map_model_ids_species(project, model_ids):
    map_model_ids_species = dict()
    train_files = [os.path.join(project.CNN_output_dir,'CNN_models_summary_k_'+str(k)+'.txt')
                           for k in range(1, project.MAXIMAL_K)]
    for train_path in train_files:
        with open(train_path) as train_file:
            for line in train_file:
                if re.match("^\s$", line) or line.startswith("#######"):
                    continue
                split_line = line.split("\t")
                train = split_line[0].split()[1]
                model_id = split_line[2].split()[1]
                if model_id not in model_ids:
                    continue
                map_model_ids_species[model_id] = train

    return map_model_ids_species


def get_sorted_models_list(project):
    model_ids = []
    entries = []
    checkpoints_folder = project.checkpoints_folder_tmp # TODO delete
    # checkpoints_folder = project.checkpoints_folder
    for filename in os.listdir(checkpoints_folder):
        if filename.endswith(".tar"):
            model_ids.append(filename[:-len(".tar")])
            entries.append((os.path.join(checkpoints_folder, filename)))
    print("len model_ids: ", len(model_ids))
    # exit()
    map_ids_species = get_map_model_ids_species(project, model_ids)
    # entries = ((os.stat(path), path) for path in entries)
    # entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))
    # # sort tar files from newest to oldest, and take only the #species newest models.
    # sorted_entries = sorted(entries, reverse=True)
    # print("sorted_entries: ", sorted_entries)
    #
    # sorted_entries = sorted_entries[:len(project.species)]
    # print("len(sorted_entries) = ", len(sorted_entries))
    # i = 0
    # for cdate, path in sorted_entries:
    #     print("cdate: ", cdate)
    #     print("path: ", path)
    #     exit()
    #     # print("time.ctime(cdate), os.path.basename(path): ", time.ctime(cdate), os.path.basename(path))
    #     best_model_validation_id = os.path.basename(path)[:-len('.tar')]
    #     print("best_model_validation_id: ", best_model_validation_id)
    #     map_model_ids[best_model_validation_id] = project.species[i]
    #     model_dir = os.path.join(checkpoints_folder, best_model_validation_id)

    #     i += 1
    # print("len(map_model_ids): ", len(map_model_ids))
    first_species = None
    sorted_list = []
    for item in map_ids_species.items():
        if project.project_name == "H3K27ac_vs_k_shuffle":
            if item[1] != "Naked_mole_rat":
                sorted_list.append(item[0])
            else:
                first_species = item[0]
        elif project.project_name == "H3K27ac_vs_negative_data":
            if item[1] != "Mouse":
                sorted_list.append(item[0])
            else:
                first_species = item[0]
        elif project.project_name == "TF_vs_k_shuffle":
            if item[1] != "Monodelphis_domestica":
                sorted_list.append(item[0])
            else:
                first_species = item[0]
        elif project.project_name == "TF_vs_negative_data":
            if item[1] != "Homo_sapiens":
                sorted_list.append(item[0])
            else:
                first_species = item[0]
        else:
            if item[1] != "Monodelphis_domestica":
                sorted_list.append(item[0])
            else:
                first_species = item[0]
    if first_species:
        print("first_species = ", first_species)
        sorted_list.insert(0, first_species)
    return sorted_list, map_ids_species


def draw_roc_curve(array_true_labels, array_prediction_scores, project, output_dir,
                   best_model_validation_id, out_file, train_species, test_species,
                   plot_ROC=True, average_auc=None):
    print ("average_auc: ", average_auc)
    if project.k:
        figure_roc_path = os.path.join(output_dir, "ROC_CNN_k_"+str(project.k)+"_"+
                                       str(best_model_validation_id) + ".pdf")
    else:
        figure_roc_path = os.path.join(output_dir, "ROC_CNN_" +

                                       str(best_model_validation_id) + ".pdf")
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x = [0, 1]
    plt.plot(x, x, 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.suptitle('ROC curve')
    data_name = " ".join(project.project_name.split("_"))
    if average_auc is None:
        fpr, tpr, thresholds = roc_curve(array_true_labels, array_prediction_scores)
        auc = roc_auc_score(array_true_labels, array_prediction_scores)
        print("auc = ", auc)
        out_file.write("auc: {0:.3f}".format(auc)+"\n")
        if not plot_ROC:
            return auc
        if project.PWM:
            motif_name = project.PWM.split("_")[0]
            plt.title('CNN, ' + data_name + ' of TF: ' + motif_name)
            plt.plot(fpr, tpr, label="AUC: {0:.3f}".format(auc))
            plt.legend(loc='best')
        else:
            train_species = " ".join(train_species.split("_"))
            if train_species == "All species 238000" or train_species == "All species 60000":
                train_species = "all species"
            if "k_shuffle" in project.project_name:
                data_name_split = data_name.split()
                new_data_name = data_name_split[0] + " " + data_name_split[1] + " k-shuffle"
                plt.title('CNN, ' + new_data_name + ', trained on ' + train_species + ', k=' + str(project.k))
            else:
                plt.title('CNN, ' + data_name + ', trained on ' + train_species)
            test_species_split = " ".join(test_species.split("_"))
            str_label = test_species_split + ', '
            plt.plot(fpr, tpr, label=str_label + "AUC: {0:.3f}".format(auc))
            fontP = FontProperties()
            fontP.set_size('small')
            plt.legend(title="Test", loc='best', prop=fontP)
    else:  # last_iteration - just plot the average auc of all tested species on the ROC figure
        ax.text(0.2, 0.1, "average AUC: {0:.3f}".format(average_auc),
                style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 3})
        auc = average_auc
    plt.savefig(figure_roc_path, format='pdf')
    print("saving figure as pdf: ", figure_roc_path)
    print("")
    return auc


def write_labels_and_scores(output_dir, best_model_validation_id, array_prediction_scores, array_true_labels):
    out_score_path = os.path.join(output_dir, "scores_" +
                                  str(best_model_validation_id) + ".txt")
    out_true_labels_path = os.path.join(output_dir, "labels_" +
                                        str(best_model_validation_id) + ".txt")
    print("write prediction scores file")
    with open(out_score_path, 'w') as out_score:
        for value in array_prediction_scores:
            out_score.write(str(value) + "\n")
    print("write labels file")
    with open(out_true_labels_path, 'w') as out_label:
        for value in array_true_labels:
            out_label.write(str(value) + "\n")


def get_test_samples_path(project, test_species):
    if project.project_name != "simulated_data":
        if project.k:
            test_x_path = os.path.join(project.samples_base_dir, test_species,
                                       project.k_let_dirs[project.k-1], 'test_X.npy')
            test_y_path = os.path.join(project.samples_base_dir, test_species,
                                       project.k_let_dirs[project.k-1], 'test_Y.npy')
            if project.project_name == "negative_data_vs_k_shuffle":
                test_x_path = []
                test_y_path = []
                for k in range(1, project.MAXIMAL_K+1):
                    test_x_path.append(os.path.join(project.samples_base_dir, test_species,
                                               project.k_let_dirs[k], 'test_X.npy'))
                    test_y_path.append(os.path.join(project.samples_base_dir, test_species,
                                               project.k_let_dirs[k], 'test_Y.npy'))

        else:
            test_x_path = os.path.join(project.samples_base_dir, test_species, 'test_X.npy')
            test_y_path = os.path.join(project.samples_base_dir, test_species, 'test_Y.npy')
    else:
        test_x_path = os.path.join(project.samples_base_dir, 'test_X.npy')
        test_y_path = os.path.join(project.samples_base_dir, 'test_Y.npy')
    return test_x_path, test_y_path


def get_scores_and_labels(result, test_samples_sequences, test_correct_labels):
    all_scores = []
    all_true_labels = []
    for sample_index in range(len(result)):
        sample_object = SampleObject(test_samples_sequences[sample_index],
                                     test_correct_labels[sample_index],
                                     True)
        true_label_int = sample_object.convert_matrix_to_label()
        y_score = result[sample_index][1]  # takes only the score of the negative label
        all_scores.append(y_score)
        all_true_labels.append(true_label_int)
    return all_scores, all_true_labels


def import_model_and_test(project, best_model_validation_id, test_species,
                          train_species, out_file=None):
    checkpoints_folder = project.checkpoints_folder_tmp  # TODO delete
    # checkpoints_folder = project.checkpoints_folder
    # model checkpoint:
    model_variables_path = os.path.join(checkpoints_folder, best_model_validation_id,
                                        best_model_validation_id)
    # model definition:
    model_def_path = os.path.join(checkpoints_folder, best_model_validation_id,
                                  best_model_validation_id + ".meta")
    # running the imported model on the test sample
    test_x_path, test_y_path = get_test_samples_path(project, test_species)
    if type(test_x_path) == list and type(test_y_path) == list:
    saver = tf.train.import_meta_graph(model_def_path)
    with tf.Session() as sess:
        saver.restore(sess, model_variables_path)
        test_set = DataSetObject(test_x_path, test_y_path)
        test_samples = test_set.get_next_batch()  # get all test labels
        test_samples_sequences, test_correct_labels = test_set.get_samples_labels(test_samples)
        result = sess.run("output:0", feed_dict={"input_x:0": test_samples_sequences,
                                                 "input_y:0": test_correct_labels,
                                                 "input_keep_prob:0": 1.0})
    all_scores, all_true_labels = get_scores_and_labels(result, test_samples_sequences,
                                                        test_correct_labels)
    if out_file:
        out_file.write('best_model_validation_id = ' + best_model_validation_id + '\n' +
                'train: ' + train_species + '\n' + 'test: ' +
                       test_species + "\n")
    # df_correct_labels_test = pd.DataFrame(data=np.array(test_correct_labels))
    array_prediction_scores = np.array(all_scores)
    array_true_labels = np.array(all_true_labels)
    # df_prediction_labels = pd.DataFrame(data=np.array(array_prediction_scores))
    return array_true_labels, array_prediction_scores

