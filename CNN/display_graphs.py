__author__ = 'Dikla Cohn'

import os
import numpy as np
import test_CNN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
import seaborn as sns

legend = True


def read_scores_and_labels_files(project, model_validation_id, test_species):
    score_path = os.path.join(project.CNN_output_dir,
                              model_validation_id,
                              test_species+"_scores_" + str(model_validation_id) + ".txt")
    true_labels_path = os.path.join(project.CNN_output_dir,
                                    model_validation_id,
                                    test_species+"_labels_" + str(model_validation_id) + ".txt")
    scores = []
    labels = []
    # print("read prediction scores file")
    with open(score_path) as score_file:
        for line in score_file:
            scores.append(float(line))
    # print("read labels file")
    with open(true_labels_path) as label_file:
        for line in label_file:
            labels.append(float(line))
    return np.array(scores),  np.array(labels)


def draw_ROC_graphs(project, model_validation_id, k, color):
    number_of_test_species = len(project.species)
    figure_roc_path = os.path.join(project.CNN_output_dir, "ROC_CNN_different_k_trained_on_human.pdf")

    # Compute ROC curve and ROC area for each test species
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(number_of_test_species):
        test_species = project.species[i]
        scores, labels = read_scores_and_labels_files(project, model_validation_id, test_species)
        fpr[i], tpr[i], _ = roc_curve(labels, scores)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(number_of_test_species)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(number_of_test_species):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= number_of_test_species
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"], color=color,
             label='k = '+str(k)+', AUC = {0:0.2f}'.format(roc_auc["macro"]),
             linewidth=3)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    x = [0, 1]
    plt.plot(x, x, 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.suptitle('ROC curve')
    project_name_split = (project.project_name).split("_")
    if project_name_split[0] == "negative":
        project_name_split[0] = "Negative"
    data_name = " ".join(project_name_split)
    data_name_split = data_name.split()
    new_data_name = data_name_split[0] + " " + data_name_split[1] + " vs. k-shuffle"
    plt.title(new_data_name + ', with different values of k')
    if legend:
        plt.legend(loc='best')
        figure_roc_path = os.path.join(project.CNN_output_dir, "ROC_CNN_different_k_trained_on_human_with_legend.pdf")
    plt.savefig(figure_roc_path, format='pdf')
    print("saving figure as pdf: ", figure_roc_path)
    print("")
    return auc


def draw_k_graph(project, model_ids):
    number_of_test_species = len(project.species)
    if project.project_name == "TF_vs_k_shuffle":
        train = "All_species_60000"
    else:
        train = "Human"
    figure_roc_path = os.path.join(project.CNN_output_dir,
                                   "AUC_CNN_different_k_trained_on_" + train + ".pdf")

    tested_on_all_ks = []
    colors = sns.color_palette("hls", number_of_test_species)

    for k in range(1, project.MAXIMAL_K + 1):
        auc = None
        tested_on_one_k = [None] * number_of_test_species
        test_results_file = os.path.join(project.CNN_output_dir,
                                         "CNN_test_output_k_" + str(k) + ".txt")
        wrong_model_id = False
        with open(test_results_file) as results_file:
            for line in results_file:
                if re.match("^\s$", line):
                    continue
                elif line.startswith("best_model_validation_id:") or \
                        line.startswith("finish test"):
                    continue
                split_line = line.split()
                # print("line = ",line)
                if line.startswith("best_model_validation_id"):
                    model_id = split_line[2]
                    if model_id not in model_ids:
                        wrong_model_id = True
                    else:
                        wrong_model_id = False
                elif line.startswith("train:") and not wrong_model_id:
                    train = split_line[1]
                    if project.project_name == "TF_vs_k_shuffle":
                        if train != "All_species_60000":
                            continue
                    else:
                        if train != "Human":
                            continue
                elif line.startswith("test:") and not wrong_model_id:
                    test = split_line[1]
                    if "All_species" in test:
                        break
                elif line.startswith("auc:") and not wrong_model_id:
                    auc = float(split_line[1])
                    wrong_model_id = False
                if auc and not wrong_model_id:
                    test_index = project.species.index(test)
                    tested_on_one_k[test_index] = auc
        tested_on_all_ks.append(tested_on_one_k)

    plots = [None for i in range((number_of_test_species))]
    x_axis = [i for i in range(1, project.MAXIMAL_K + 1)]
    # y_axis = [None for i in range(len(species_indices_map))]

    fig1 = plt.figure(0)
    plt.grid(True)
    tested_on_all_ks_array = np.array(tested_on_all_ks)

    for species_index in range((number_of_test_species)):
        test_species = project.species[species_index]
        results_list = tested_on_all_ks_array[:, species_index]
        y_axis = results_list
        All_aucs = []
        for i in range(project.MAXIMAL_K):
            All_aucs.append(results_list[i])
        test_species_split = " ".join(test_species.split("_"))
        plots[species_index], = plt.plot(x_axis, All_aucs, color=colors[species_index],
                                         label=test_species_split)
        ax = fig1.add_subplot(111)
        # ax.scatter(x_axis, All_aucs)
        # for i, txt in enumerate(All_aucs):
        #     ax.annotate(("{0:.3f}".format(float(txt))), (x_axis[i], All_aucs[i]))
        species_index += 1

    # plt.axis([1, 5, 0, 1])
    ax.set_xlim(1, project.MAXIMAL_K)
    # plt.xticks(x_axis, ["1", "2", "3", "4"])
    plt.xlabel('Shuffling parameter k', fontsize=13)
    plt.ylabel('AUC', fontsize=13)
    plt.suptitle('AUC results as a function of the shuffling parameter k', fontsize=10)
    if project.project_name == "negative_data_vs_k_shuffle":
        plt.title("Negative data vs. k-shuffle", fontsize=10)
    elif project.project_name == "TF_vs_k_shuffle":
        plt.title("TF vs. k-shuffle", fontsize=10)

    fontP = FontProperties()
    fontP.set_size('small')
    if legend:
        plt.legend(title="Test", loc='best', prop=fontP)
        if project.project_name == "TF_vs_k_shuffle":
            train = "All_species_60000"
        else:
            train = "Human"
        figure_roc_path = os.path.join(project.CNN_output_dir,
                                       "AUC_CNN_different_k_trained_on_"+train+"_with_legend.pdf")

    plt.savefig(figure_roc_path, format='pdf')
    print("saving figure as pdf: ", figure_roc_path)




def main():
    os.system("module load tensorflow")
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'display_graphs.py')
    model_ids = []
    checkpoints_folder = project.checkpoints_folder_tmp
    for filename in os.listdir(checkpoints_folder):
        if filename.endswith(".tar"):
            model_ids.append(filename[:-len(".tar")])
    print("len model_ids: ", len(model_ids))
    draw_k_graph(project, model_ids)
    exit()

    ###############################################################################
    colors = sns.color_palette("hls", project.MAXIMAL_K)
    sorted_ids = [None] * project.MAXIMAL_K
    for model_validation_id in model_ids:
        split_id = model_validation_id.split("_")
        if len(split_id) == 4:
            print("split_id: ", split_id)
            k = int(model_validation_id.split("_")[2])
        else:
            k = 4
        sorted_ids[k-1] = model_validation_id
    print("sorted_ids: ", sorted_ids)

    for k in range(1, project.MAXIMAL_K+1):
        model_validation_id = sorted_ids[k-1]
        draw_ROC_graphs(project, model_validation_id, k, colors[k-1])

    print("End!!!")


if __name__ == "__main__":
    main()


