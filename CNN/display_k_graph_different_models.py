__author__ = 'Dikla Cohn'

import os
from Project import Project
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import sys
from matplotlib.font_manager import FontProperties

base_path = os.path.dirname(os.path.abspath(__file__))

legend = False
titles = False
train_on_human = True
train_on_all_samples = False
train_on_dog = False

maximal_k = 9

def draw_k_graph(figure_path, train_species, results_path):
    x_axis = [i for i in range(1, maximal_k + 1)]
    fig1 = plt.figure(0)
    plt.grid(True)
    y = [0.5] * maximal_k
    print("y = ", y)
    plt.plot(x_axis, y, 'k--', linewidth="2.0")
    tested_on_all_ks_all_projects = []
    project_names = []
    tested_on_all_ks_one_project = []
    with open(results_path) as results_file:
        for line in results_file:
            if re.match("^\s$", line):
                continue
            elif not line.startswith("k = "):
                project_name = line
                print("project_name = ", project_name)
                project_names.append(project_name)
                if len(tested_on_all_ks_one_project) == maximal_k:
                    tested_on_all_ks_all_projects.append(tested_on_all_ks_one_project)
                tested_on_all_ks_one_project = []

            else:
                split_line = line.split()
                k = int(split_line[2])
                auc = float(split_line[6])
                print("k = ", k, " , auc = ", auc)
                tested_on_all_ks_one_project.append(auc)
    tested_on_all_ks_all_projects.append(tested_on_all_ks_one_project)
    for i in range(len(tested_on_all_ks_all_projects)):
        project_name = project_names[i]
        tested_on_all_ks_one_project = tested_on_all_ks_all_projects[i]
        if "TF" in project_name:
            label_project = "TF peaks vs. k-shuffled"
        elif "negative_data" in project_name:
            label_project = "Non-enhancers vs. k-shuffled"
        elif "H3K27ac" in project_name:
            label_project = "Enhancers vs. k-shuffled"
        plt.plot(x_axis, tested_on_all_ks_one_project, label=label_project, linewidth="2.0")

        ax = fig1.add_subplot(111)
        ax.set_xlim(1, maximal_k)
        plt.xlabel('Shuffling parameter k', fontsize=20)
        plt.ylabel('Mean AUC', fontsize=20)
        # labels = [str(i) for i in range(1, maximal_k+1)]
        # plt.xticks(x_axis, labels)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(17)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(17)

        if titles:
            plt.suptitle('Classification results as a function of the shuffling parameter k', fontsize=12)
            train_species_new = " ".join(train_species.split("_"))
            plt.title("Comparison between models, trained on "+train_species_new, fontsize=12)

        # fontP = FontProperties()
        # fontP.set_size('small')
        if legend:
            plt.legend(loc='best', fontsize=18)
            new_figure_path = (figure_path[:-len(".pdf")]) + "_with_legend.pdf"
        else:
            new_figure_path = figure_path
        plt.savefig(new_figure_path, format='pdf')
        print("saving figure as pdf: ", new_figure_path)


def draw_k_graph_and_write_results(project, model_ids, figure_path, train_species, out_results_all_models):
    number_of_test_species = len(project.species) - 2
    print("number_of_test_species = ", number_of_test_species)
    tested_on_all_ks = []
    out_results_all_models.write(project.project_name+"\n")
    for k in range(1, project.MAXIMAL_K + 1):
        auc = None
        tested_on_one_k = [None] * number_of_test_species
        sum_tested_on_one_k = 0
        test_results_file = os.path.join(project.CNN_output_dir,
                                         "CNN_test_output_k_" + str(k) + ".txt")
        wrong_model_id = True

        with open(test_results_file) as results_file:
            for line in results_file:
                if re.match("^\s$", line):
                    continue
                elif line.startswith("finish test") or line.startswith("average_auc_returned:"):
                    continue
                elif None not in tested_on_one_k:
                    print("tested_on_one_k is full")
                    break
                # else:
                    # print("tested_on_one_k", tested_on_one_k)
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
                    if train != train_species:
                        continue
                elif line.startswith("test:") and not wrong_model_id:
                    test = split_line[1]
                    if "All_species" in test:
                        wrong_model_id = True
                        continue
                elif line.startswith("auc:") and not wrong_model_id:
                    auc = float(split_line[1])
                    wrong_model_id = False
                    if auc and not wrong_model_id:
                        test_index = project.species.index(test)
                        tested_on_one_k[test_index] = auc
                        sum_tested_on_one_k += auc
                        # print("test = ", test)
        average_auc_one_k = sum_tested_on_one_k / number_of_test_species
        print("k = ", k, " , average_auc = ", average_auc_one_k)
        out_results_all_models.write("k = "+ str(k)+ " , average_auc = "+ str(average_auc_one_k) + "\n")
        tested_on_all_ks.append(average_auc_one_k)
        # tested_on_all_ks.append(tested_on_one_k)

    # plots = [None for i in range(number_of_test_species)]
    x_axis = [i for i in range(1, project.MAXIMAL_K + 1)]
    fig1 = plt.figure(0)
    plt.grid(True)

    if "TF" in project.project_name:
        label_project = "TF peaks vs. k-shuffle"
    elif "negative_data" in project.project_name:
        label_project = "Non-enhancer data vs. k-shuffle"
    elif "H3K27ac":
        label_project = "Enhancers vs. k-shuffle"
    plt.plot(x_axis, tested_on_all_ks,  label=label_project)
    ax = fig1.add_subplot(111)
    ax.set_xlim(1, project.MAXIMAL_K)
    plt.xlabel('Shuffling parameter k', fontsize=13)
    plt.ylabel('AUC average over all test species', fontsize=13)
    plt.suptitle('Classification results as a function of the shuffling parameter k', fontsize=12)
    train_species_new = " ".join(train_species.split("_"))
    plt.title("Comparison between models, trained on "+train_species_new, fontsize=12)

    fontP = FontProperties()
    fontP.set_size('small')
    if legend:
        plt.legend(title="Model", loc='best', prop=fontP)
        new_figure_path = (figure_path[:-len(".pdf")])+"_with_legend.pdf"
    else:
        new_figure_path = figure_path

    plt.savefig(new_figure_path, format='pdf')
    print("saving figure as pdf: ", new_figure_path)


def main():
    os.system("module load tensorflow")
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 display_k_graph_different_models.py <project_name_1> <project_name_2> <project_name_3>\n")
    project_name_1 = sys.argv[1]
    project_name_2 = sys.argv[2]
    project_name_3 = sys.argv[3]
    base_path_projects = base_path[:-len("CNN/")]
    project_1 = Project(project_name_1, base_path_projects)
    project_2 = Project(project_name_2, base_path_projects)
    project_3 = Project(project_name_3, base_path_projects)



    for project in [project_1, project_2, project_3]:
        model_ids = []
        checkpoints_folder = project.checkpoints_folder_tmp
        for filename in os.listdir(checkpoints_folder):
            if project.project_name.startswith("H3K27ac"):
                path = os.path.join(checkpoints_folder, filename)
                if os.path.exists(path) and os.path.isdir(path):
                    model_ids.append(filename)
            else:
                if filename.endswith(".tar"):
                    model_ids.append(filename[:-len(".tar")])


        print("len model_ids: ", len(model_ids))

        if project.project_name == "" :
            sorted_ids = [None] * project.MAXIMAL_K
            for model_validation_id in model_ids:
                split_id = model_validation_id.split("_")
                if len(split_id) == 4:
                    print("split_id: ", split_id)
                    k = int(model_validation_id.split("_")[2])
                else:
                    k = 4
                sorted_ids[k - 1] = model_validation_id
            print("sorted_ids: ", sorted_ids)
        else:
            sorted_ids = model_ids

        if train_on_human:
            if "Homo_sapiens" in project.species:
                train = "Homo_sapiens"
            elif "Human" in project.species:
                train = "Human"
            file_name = "Human"
        elif train_on_all_samples:
            if "All_species_60000" in project.species:
                train = "All_species_60000"
            elif "All_species_238000" in project.species:
                train = "All_species_238000"
            file_name = "All_species"

        elif train_on_dog:
            if "Canis_familiaris" in project.species:
                train = "Canis_familiaris"
            elif "Dog" in project.species:
                train = "Dog"
            file_name = "Dog"
        figure_path = os.path.join(project_1.all_projects_base_bath,
                                   "AUC_CNN_different_k_different models_trained_on_" +file_name +".pdf")
        results_path = os.path.join(project_1.all_projects_base_bath,
                                   "AUC_results_CNN_different_k_different models_trained_on_" +file_name +".txt")
        # with open(results_path, "a+") as out_results_all_models:
        #     draw_k_graph_and_write_results(project, sorted_ids, figure_path, train, out_results_all_models)

    draw_k_graph(figure_path, train, results_path)



    print("End!!!")


if __name__ == "__main__":
    main()


