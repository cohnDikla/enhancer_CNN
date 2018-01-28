import os
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, base_path+"/CNN/")
import test_CNN
sys.path.insert(0, base_path+"/SVM/")
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
create a ROC figure with results of 3 models: PSSM with JASPAR motif,
                                              PSSM with denovo motif,
                                              and CNN.
running on the same project (dataset).
"""

PSSM_models = ["denovo", "JASPAR"]

def find_best_kernel_and_kmer_size(project):
    best_k = None
    best_kernel = None
    max_auc = -1
    output_dir = os.path.join(project.SVM_output_dir,
                              project.distribution_samples_center_dir,
                              project.PWM)
    results_path = os.path.join(output_dir,
                                "results_SVM_different_kernels_" + project.PWM + "_" +
                                project.distribution_samples_center_dir + ".txt")
    with open(results_path) as results_file:
        for line in results_file:
            if line.endswith("\n"):
                line = line.strip("\n")
            if line.endswith(" kernel"):
                kernel = line[:-len(" kernel")]
            elif line.startswith("k-mer size:"):
                kmer_size = int(line[len("k-mer size:")])
            elif line.startswith("auc: "):
                auc = float(line.split()[1])
                if auc > max_auc:
                    max_auc = auc
                    best_kernel = kernel
                    best_k = kmer_size
    print("best_kernel = ", best_kernel)
    print("best_k = ", best_k)
    return best_kernel, best_k


def read_scores_and_labels_files(model_dir, project, best_model_validation_id=None,
                                 pssm_model=None, pr=None):
    if best_model_validation_id:
        score_path = os.path.join(model_dir, "scores_" + str(best_model_validation_id) + ".txt")
        true_labels_path = os.path.join(model_dir, "labels_" + str(best_model_validation_id) + ".txt")
    else:
        if pr:
            if pr == "with_prior":
                score_path = os.path.join(model_dir, "scores_PSSM_"+pr+"_" + pssm_model + "_" +
                                          project.distribution_samples_center_dir + ".txt")
                true_labels_path = os.path.join(model_dir, "labels_PSSM_"+pr+"_" + pssm_model + "_" +
                                                project.distribution_samples_center_dir + ".txt")
            else:
                score_path = os.path.join(model_dir, "scores_PSSM_" + pssm_model + "_" +
                                          project.distribution_samples_center_dir + ".txt")
                true_labels_path = os.path.join(model_dir, "labels_PSSM_" + pssm_model + "_" +
                                                project.distribution_samples_center_dir + ".txt")
    scores = []
    labels = []
    print("read prediction scores file")
    with open(score_path) as score_file:
        for line in score_file:
            scores.append(float(line))
    print("read labels file")
    with open(true_labels_path) as label_file:
        for line in label_file:
            labels.append(float(line))
    return scores, labels


def add_roc_curve(labels, scores, fig, model_label):
    print("run roc_curve:")
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc = roc_auc_score(labels, scores)
    print("auc = ", auc)
    print("\n\n")
    ax = fig.add_subplot(111)
    if model_label.startswith("Gold standard"):
        plt.plot(fpr, tpr, label=model_label+"{0:.2f}".format(auc), linewidth=2.0, linestyle="--",
                color="gold")
    else:
        plt.plot(fpr, tpr, label=model_label + "{0:.2f}".format(auc), linewidth=2.0)


def main():
    fig = plt.figure(1)
    x = [0, 1]
    plt.plot(x, x, 'k--')
    project = test_CNN.get_project_and_check_arguments(sys.argv, "roc_comparison.py")
    TF_name = project.PWM.split("_")[0]   # CEBPA for example
    figure_roc_path = os.path.join(project.basic_output_dir,
                                   "ROC_comparison_between_models_" +
                                   project.distribution_samples_center_dir+"_sigma_" + str(project.sigma) + ".pdf")

    # CNN:
    sorted_models_list, map_model_ids = test_CNN.get_sorted_models_list(project)
    index_train_species = 0
    best_model_validation_id = sorted_models_list[index_train_species]
    train_species = map_model_ids[best_model_validation_id]
    print("train_species: ", train_species)
    model_dir = test_CNN.create_directories(project, best_model_validation_id)
    CNN_scores, CNN_labels = read_scores_and_labels_files(model_dir, project, best_model_validation_id=best_model_validation_id)
    model_label = "CNN: , AUC: "
    add_roc_curve(CNN_labels, CNN_scores, fig, model_label)



    # PSSM models:
    for pssm_model in PSSM_models:  # ["denovo", "JASPAR"]

        for pr in ["with_prior", "without_prior"]:
            dir_name = "CEBPA_"+pssm_model

            PSSM_output_dir = os.path.join(project.PSSM_output_dir, project.distribution_samples_center_dir,
                                           dir_name)
            PSSM_scores, PSSM_labels = read_scores_and_labels_files(PSSM_output_dir, project, pssm_model=pssm_model, pr=pr)
            if pr == "with_prior":
                # model_label = "PSSM, " + pssm_model + ", with location prior, AUC: "
                if pssm_model == "denovo":
                    model_label = "Homer de-novo motif w/ location prior, AUC: "
                elif pssm_model == "JASPAR":
                    model_label = "Gold standard model - True motif w/ location prior, AUC: "
            else:
                # model_label = "PSSM, " + pssm_model + ", AUC: "
                if pssm_model == "denovo":
                    model_label = "Homer de-novo motif w/o location prior, AUC: "
                elif pssm_model == "JASPAR":
                    model_label = "True motif w/o location prior, AUC: "
            add_roc_curve(PSSM_labels, PSSM_scores, fig, model_label)


    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.suptitle('ROC curve')
    data_name = " ".join(project.project_name.split("_"))
    # plt.title('Comparison between models, ' + data_name + ' of single TF: ' + TF_name)
    plt.legend(loc='best')
    plt.savefig(figure_roc_path, format='pdf')
    print("saving figure: ", figure_roc_path, "\n\n")





if __name__ == "__main__":
    main()
    print("End!")




