__author__ = 'Dikla Cohn'

import os
base_path = os.path.dirname(os.path.abspath(__file__))
print("base_path = ", base_path)
base_path = base_path[:-len("/PSSM_straw_man_model")]
import sys
sys.path.insert(0, base_path+'/CNN/')
from Project import Project
import data_handle
import math
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats

"""
PSSM straw man model:
Calculates PSSM score for each sample, according to the given PWM.
Finds the best threshold, that gives the best accuracy.
Takes samples both as forward form and as the reverse complement of the sample,
so the actual number of samples id doubled.
"""

BACKGROUND = 1/4
# prior normal distribution of motif's location:
mu = data_handle.SAMPLE_LENGTH / 2
sigma = 30

def check_arguments():
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python3 straw_man_model.py <project_name_PWM_name> [<k=None> or <normal_distribution_string>] \n")
    k = None
    is_normal_distribution = False
    project_name = sys.argv[1]
    if len(sys.argv) == 3:
        if sys.argv[2].isdigit():
            k = int(sys.argv[2])
            print("k = ", k)
        else:
            is_normal_distribution = bool(sys.argv[2])  # this is True
    if k:
        project = Project(project_name, base_path, k=k)
    else:
        project = Project(project_name, base_path, normal_distribution=is_normal_distribution)
    return project


def calculate_PSSM(samples_file_path, PWM_path, with_prior=False, is_denovo=False):
    """
    Calculate PSSM for all given samples and their reverse complement, according to the given
    PWM matrix.
    """
    if with_prior:
        pdfs = [scipy.stats.norm(mu, sigma).pdf(val) for val in range(data_handle.SAMPLE_LENGTH)]
        max_pdf = max(pdfs)
        # print("max_pdf: ", max_pdf)
        # new_pdfs = [(val / max_pdf) for val in pdfs]
    PSSM_vector_all_samples = []
    # if species_index == 0:
    #     out = open(out_path, 'w+')
    with open(samples_file_path) as samples_file:
        if not is_denovo:
            PWM = data_handle.create_frequency_matrix_with_pseudo_counts(PWM_path, is_denovo=is_denovo)
        else:
            PWM = data_handle.create_frequency_matrix_with_pseudo_counts(PWM_path, is_denovo=is_denovo)
        motif_length = len(PWM[0])
        rev_comp_PWM = data_handle.get_reverse_complement_matrix(PWM)
        # print("motif length = ", motif_length)
        line_counter = 0
        for line in samples_file:
            line_counter+=1
            # if line_counter >= 1000:  # TODO delete
            #     break
            original_sample = line[:data_handle.SAMPLE_LENGTH]
            # rev_comp_sample = data_handle.get_reverse_complement(original_sample)
            for sample in [original_sample]:
                index_in_line = -1
                PSSM_vector_one_sample = []
                for j in range(data_handle.SAMPLE_LENGTH):
                    # base = sample[j]
                    index_in_line += 1
                    # not calculate PSSM for the first 11 bases:
                    if index_in_line < motif_length:
                        continue
                    # set window to be such that the current index_in_line is the window's center
                    if motif_length % 2 == 1:  # motif_length is odd
                        window = sample[index_in_line - int(motif_length/2):
                                        index_in_line + int(motif_length/2)+1]
                    else:  # motif_length is even
                        window = sample[index_in_line - int(motif_length / 2):
                                        index_in_line + int(motif_length / 2)]
                    # print("window = ", window)
                    product_numerator = 1
                    product_denominator = 1
                    # product_rev = 1
                    for index_in_window in range(len(window)):
                        base_in_window = window[index_in_window]
                        base_index = data_handle.bases_map[base_in_window]
                        frequency = PWM[base_index, index_in_window]  # PWM_i
                        # frquency_rev = rev_comp_PWM[base_index, index_in_window]  # PWM_i
                        product_numerator *= frequency
                        product_denominator *= BACKGROUND
                        # product_rev *= frquency_rev
                        # print("frequency = ", frequency)
                    if with_prior:
                        # product *= new_pdfs[index_in_line]  # TODO ???
                        product_numerator *= pdfs[index_in_line]  # TODO ???
                        product_denominator *= (1/data_handle.SAMPLE_LENGTH)

                    # product_rev = product_rev / BACKGROUND
                    PSSM = math.log(product_numerator/product_denominator, 2)  # log with base 2
                    # if with_prior:
                    #     # product *= new_pdfs[index_in_line]  # TODO ???
                    #     PSSM *= (pdfs[index_in_line] / (1/data_handle.SAMPLE_LENGTH))  # TODO ???
                    # PSSM_rev = math.log(product_rev, 2)  # log with base 2
                    # max_score = max(PSSM, PSSM_rev)
                    # PSSM_vector_one_sample.append(max_score)
                    PSSM_vector_one_sample.append(PSSM)

                max_PSSM = max(PSSM_vector_one_sample)
                # print("arg max:" , np.argmax(PSSM_vector_one_sample))
                PSSM_vector_all_samples.append(max_PSSM)
                # if species_index == 0:
                #     out.write(str(max_PSSM)+"\n")
    # if species_index == 0:
    #     out.close()
    return PSSM_vector_all_samples


# def calculate_PSSM_with_prior(samples_file_path, PWM_path):
#     """
#     Calculate PSSM with prior of normal distribution, with mean = mu stddev = sigma.
#     Returns the PSSM score for all given samples and their reverse complement,
#     according to the given PWM matrix.
#     """
#     pdfs = [scipy.stats.norm(mu, sigma).pdf(val) for val in range(data_handle.SAMPLE_LENGTH)]
#     max_pdf = max(pdfs)
#     # print("max_pdf: ", max_pdf)
#     new_pdfs = [(val/max_pdf) for val in pdfs]
#     # print("np.argmax(pdfs): ", np.argmax(pdfs))
#     # print("min(pdfs): ", min(pdfs))
#     # print("np.argmin(pdfs): ", np.argmin(pdfs))
#     # print("sum(pdfs): ", sum(pdfs))
#     # avg = sum(pdfs) / len(pdfs)
#     # print("avg: ", avg)
#     # cdfs = [scipy.stats.norm(mu, sigma).cdf(val) for val in range(data_handle.SAMPLE_LENGTH)]
#     # print("max(cdfs): ", max(cdfs))
#     # print("np.argmax(cdfs): ", np.argmax(cdfs))
#     # print("min(cdfs): ", min(cdfs))
#     # print("np.argmin(cdfs): ", np.argmin(cdfs))
#     # print("sum(cdfs[:data_handle.SAMPLE_LENGTH/2])): ", sum(cdfs[:int(data_handle.SAMPLE_LENGTH/2)]))
#     # avg = sum(cdfs) / len(cdfs)
#     # print("avg: ", avg)
#
#     PSSM_vector_all_samples = []
#     # if species_index == 0:
#     #     out = open(out_path, 'w+')
#     with open(samples_file_path) as samples_file:
#         PWM = data_handle.create_frequency_matrix_with_pseudo_counts(PWM_path)
#         motif_length = len(PWM[0])
#         # print("motif length = ", motif_length)
#         line_counter = 0
#         for line in samples_file:
#             line_counter+=1
#             # if line_counter >= 1000:  # TODO delete
#             #     break
#             original_sample = line[:data_handle.SAMPLE_LENGTH]
#             # rev_comp_sample = data_handle.get_reverse_complement(original_sample)
#             # for sample in [original_sample, rev_comp_sample]: # TODO ???
#             for sample in [original_sample]:
#                 index_in_line = -1
#                 PSSM_vector_one_sample = []
#                 # current_max = -(math.inf)
#                 for j in range(data_handle.SAMPLE_LENGTH):
#                     # base = sample[j]
#                     index_in_line += 1
#                     # not calculating PSSM for the first and last <motif_length/2> bases:
#                     if (index_in_line < int(motif_length/2)) or \
#                             (index_in_line > (data_handle.SAMPLE_LENGTH-int(motif_length/2)+1)):
#                         continue
#                     # set window to be such that the current index_in_line is the window's center
#                     window = sample[index_in_line - int(motif_length/2):
#                                     index_in_line + int(motif_length/2)+1]
#                     # print("window = ", window)
#                     product = 1
#                     for index_in_window in range(len(window)):
#                         base_in_window = window[index_in_window]
#                         base_index = data_handle.bases_map[base_in_window]
#                         frequency = PWM[base_index, index_in_window]  # PWM_i
#                         product *= frequency
#                         # print("frequency = ", frequency)
#                     product *= new_pdfs[index_in_line]  # TODO ???
#                     product = product / BACKGROUND
#                     PSSM = math.log(product, 2)  # log with base 2
#                     # PSSM_with_prior = PSSM * pdfs[index_in_line]
#                     # if index_in_line > data_handle.SAMPLE_LENGTH:
#                     #     abs_difference = index_in_line - data_handle.SAMPLE_LENGTH
#                     #     index_in_cdf = data_handle.SAMPLE_LENGTH - abs_difference
#                     # else:
#                     #     index_in_cdf = index_in_line
#                     # PSSM *= new_pdfs[index_in_line]  # TODO ???
#                     PSSM_vector_one_sample.append(PSSM)
#                 max_PSSM = max(PSSM_vector_one_sample)
#                 PSSM_vector_all_samples.append(max_PSSM)
#
#                 # if species_index == 0:
#                 #     out.write(str(max_PSSM)+"\n")
#
#     # if species_index == 0:
#     #     out.close()
#
#     return PSSM_vector_all_samples
#

def main():
    project = check_arguments()
    output_dir = os.path.join(project.PSSM_output_dir,
                              project.distribution_samples_center_dir, project.PWM)
    figure_roc_path = os.path.join(output_dir, "ROC_PSSM_" + "JASPAR" + "_" +
                                   project.distribution_samples_center_dir + ".pdf")
    if project.PWM == "denovo":
        samples_dir = os.path.join(project.base_dir_data_path,
                                            project.distribution_samples_center_dir,
                                            project.PWM, "samples")
        Homer_dir = os.path.join(samples_dir, "Homer_denovo_motifs")
        if project.project_name != "simulated_data":
            Homer_dir = os.path.join(Homer_dir, species_name)
        PWM_path = os.path.join(Homer_dir, "motifResults", "homerResults", "motif1.motif")
        is_denovo = True
    else:
        PWM_path = "/cs/cbio/dikla/projects/motifs/" + project.PWM + "_pfm_new.txt"
        is_denovo = False
    print("start PSSM calculation, project: "+project.project_name+", is_normal_distribution = "
           + str(project.normal_distribution) + ", k = ", str(project.k))
    print("PWM is : ", project.PWM)
    # for i in range(NUMBER_OF_VERTEBRATES): # TODO uncomment
    i = 0 # only Cfam # TODO delete
    print("start species: ", project.species[i])
    if project.project_name.startswith("simulated_data"):

        data_sigma_dir = os.path.join(project.project_base_path,
                                      "data", project.distribution_samples_center_dir,
                                      project.PWM, "sigma_" + str(sigma), "samples")

        positive_samples_dir = data_sigma_dir
        negative_samples_dir = data_sigma_dir
        if is_denovo:
            positive_samples_dir = os.path.join(project.base_dir_data_path,
                                                project.distribution_samples_center_dir,
                                                "project.PWM", "samples")
            negative_samples_dir = positive_samples_dir
    else:
        if is_denovo:
            positive_samples_dir = os.path.join(project.base_dir_data_path,
                                                project.distribution_samples_center_dir,
                                                project.PWM, "samples")
            negative_samples_dir = positive_samples_dir
        else:
            positive_samples_dir = os.path.join(project.text_samples_base_dir, project.species[i])
            if project.k:
                negative_samples_dir = os.path.join(project.text_samples_base_dir, project.species[i],
                                                project.k_lets_dirs[project.k-1])
            else:
                negative_samples_dir = positive_samples_dir
    positive_samples_file_path = os.path.join(positive_samples_dir, "positive_samples")
    negative_samples_file_path = os.path.join(negative_samples_dir,  "negative_samples")
    for PSSM_model in ["PSSM", "PSSM with prior"]:
        # if PSSM_model == "PSSM": # TODO delete
        #     continue
        PSSM_model_name = "_".join(PSSM_model.split())
        results_path = os.path.join(output_dir, "results_" + PSSM_model_name + "_" + "JASPAR" + "_" +
                                    project.distribution_samples_center_dir + ".txt")
        with open(results_path, 'w') as results_out:
            results_out.write("species: " + project.species[i] + "\n")
            results_out.write("start calculation of model: " + PSSM_model + "\n")
            if project.k:
                results_out.write("k = " + str(project.k) + "\n")
            if PSSM_model == "PSSM with prior":
                PSSM_vector_positive_samples = calculate_PSSM(positive_samples_file_path,
                                                              PWM_path, True, is_denovo)
                PSSM_vector_negative_samples = calculate_PSSM(negative_samples_file_path,
                                                              PWM_path, True, is_denovo)
            else:
                PSSM_vector_positive_samples = calculate_PSSM(positive_samples_file_path, PWM_path, is_denovo=is_denovo)
                PSSM_vector_negative_samples = calculate_PSSM(negative_samples_file_path, PWM_path, is_denovo=is_denovo)

            sorted_PSSM_vector_positive_samples = sorted(PSSM_vector_positive_samples, reverse=True)
            sorted_PSSM_vector_negative_samples = sorted(PSSM_vector_negative_samples, reverse=True)
            number_of_positive_samples = len(sorted_PSSM_vector_positive_samples)
            print("number_of_positive_samples = ", number_of_positive_samples)
            number_of_negative_samples = len(sorted_PSSM_vector_negative_samples)
            print("number_of_negative_samples = ", number_of_negative_samples)
            all_samples_vector = []
            for PSSM_value in sorted_PSSM_vector_positive_samples:
                all_samples_vector.append((PSSM_value, 1))
            for PSSM_value in sorted_PSSM_vector_negative_samples:
                all_samples_vector.append((PSSM_value, 0))

            sorted_vector_all_samples = sorted(all_samples_vector, key=lambda item: item[0], reverse=True)
            number_of_samples = len(sorted_vector_all_samples)
            print("number_of_samples = ", number_of_samples)
            results_out.write("number_of_samples = "+str(number_of_samples)+"\n")
            min_sorted_vector_all_samples = sorted_vector_all_samples[-1][0]
            max_sorted_vector_all_samples = sorted_vector_all_samples[0][0]
            print("min_sorted_vector_all_samples = ", min_sorted_vector_all_samples)
            print("max_sorted_vector_all_samples = ", max_sorted_vector_all_samples)
            results_out.write("min scores all samples: "+str(min_sorted_vector_all_samples)+"\n")
            results_out.write("max scores all samples: "+str(max_sorted_vector_all_samples)+"\n")
            print("run roc_curve:")
            results_out.write("run roc_curve:\n")
            sorted_vector_all_samples_array = np.array(sorted_vector_all_samples)
            sorted_vector_all_samples_array_score_only = np.array(sorted_vector_all_samples_array[:, 0])
            sorted_vector_all_samples_array_label_only = np.array(sorted_vector_all_samples_array[:, 1])
            print("write scores file")
            out_score_path = os.path.join(output_dir, "scores_" + PSSM_model_name + "_" +
                                          "JASPAR" + "_" + project.distribution_samples_center_dir + ".txt")
            out_true_labels_path = os.path.join(output_dir, "labels_" + PSSM_model_name + "_" +
                                                "JASPAR" + "_" + project.distribution_samples_center_dir + ".txt")
            with open(out_score_path, 'w') as out_score:
                for value in sorted_vector_all_samples_array_score_only:
                    out_score.write(str(value)+"\n")
            print("write labels file")
            with open(out_true_labels_path, 'w') as out_label:
                for value in sorted_vector_all_samples_array_label_only:
                    out_label.write(str(int(value))+"\n")

            true_labels_array = np.array(sorted_vector_all_samples_array_label_only)
            fpr, tpr, thresholds = roc_curve(true_labels_array, sorted_vector_all_samples_array_score_only,
                                             pos_label=1)
            auc = roc_auc_score(true_labels_array, sorted_vector_all_samples_array_score_only)
            # specificity (SPC) is true negative rate, and it equals 1-FPR
            # true_negative_rates = [(1-fp) for fp in fpr]
            # Positive likelihood ratio is TPR/FPR
            pos_likelihood_ratio = []
            accuracies = []
            index = 0
            for fp in fpr:
                tp = tpr[index]
                if fp > 0: # could be zero, at the minimal fpr
                    num_of_TP = tp * number_of_positive_samples
                    ratio = (tp / fp)  # since  number_of_positive_samples ~= number_of_negative_samples
                    pos_likelihood_ratio.append(ratio)
                    accuracy = (num_of_TP+number_of_negative_samples-(num_of_TP/ratio)) / number_of_samples
                    accuracies.append(accuracy)
                index += 1
            print("auc = ", auc)
            results_out.write("auc: " + str(auc) + "\n")
            max_accuracy = max(accuracies)
            print("max_accuracy = ", max_accuracy)
            results_out.write("max accuracy: " + str(max_accuracy) + "\n")
            max_accuracy_index = accuracies.index(max_accuracy)
            # print("max accuracy index: ", max_accuracy_index)
            threshold_of_max_accuracy = thresholds[max_accuracy_index]
            print("threshold_of_max_accuracy: ", threshold_of_max_accuracy)
            results_out.write("threshold of max Accuracy: " + str(threshold_of_max_accuracy) + "\n\n\n")
            print("\n")
            plt.plot(fpr, tpr)
            plt.plot(fpr, tpr, label=PSSM_model+", AUC: {0:.3f}".format(auc))


    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x = [0, 1]
    plt.plot(x, x, 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.suptitle('ROC curve')
    motif_name = project.PWM.split("_")[0]
    data_name = " ".join(project.project_name.split("_"))
    plt.title('PSSM straw man model, ' + data_name + ' of TF: ' + motif_name)
    plt.legend(loc='best')
    plt.savefig(figure_roc_path,  format='pdf')
    print("saving figure: ", figure_roc_path)
    print("end! :)")

if __name__ == "__main__":
    main()
