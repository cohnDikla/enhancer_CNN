"""
A script for reading pfm txt files of motifs from JASPAR,
creating a new txt file with integer counts only,
and creating a logo image of the motif.
"""
import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path[:-len('/motifs')]+'/CNN/')
import data_handle
from subprocess import call


CEBPA_JASPAR = True
HNF4A_JASPAR = False

BASES_NUMBER = 4


if CEBPA_JASPAR:
    motif_name = "CEBPA_JASPAR"
elif HNF4A_JASPAR:
    motif_name = "HNF4A_JASPAR"

PWM_path = "/cs/cbio/dikla/projects/motifs/"+motif_name+".pfm.txt"
script_create_images = "~tommy/bin/makefig.pl -nonumbers"
output_folder_path = "/cs/cbio/dikla/projects/motifs"

bases = ["A", "C", "G", "T"]

def get_motif_logo(output_folder_path):
    """
    save the PWMs / filters as pdf logo images
    :param output_folder_path
    """
    out_text_file_path = os.path.join(output_folder_path, motif_name + "_pfm_new.txt")
    if not os.path.exists(out_text_file_path):
        with open(out_text_file_path, 'w') as text_out_file:
            array_pfm = data_handle.read_PWM_from_file(PWM_path)
            for base_index in range(BASES_NUMBER):
                counts_one_base = array_pfm[base_index]
                for count in counts_one_base:
                    text_out_file.write(str(count)+"\t")
                text_out_file.write("\n")

    output_image_path = os.path.join(output_folder_path, motif_name + ".pdf")
    script = "cat " + out_text_file_path + " | " + script_create_images + " | " + \
             "/usr/bin/convert eps:- " + output_image_path
    call(script, shell=True)


def pretty_print_of_PWM(array_PWM_with_pseudo_counts):
    out_text_file_path = os.path.join(output_folder_path, motif_name + "_pretty_print_pfm_with_pseudo_counts.txt")
    with open(out_text_file_path, 'w') as text_out_file:
        text_out_file.write("\n")
        for base_index in range(BASES_NUMBER):
            text_out_file.write("\t" + bases[base_index] + " [ ")
            counts_one_base = array_PWM_with_pseudo_counts[base_index]
            count_index = 0
            for count in counts_one_base:
                if count_index < len(counts_one_base) - 1:
                    text_out_file.write("{0:.3f}".format(count) + "\t")
                else:
                    text_out_file.write("{0:.3f}".format(count))
                count_index += 1
            text_out_file.write(" ]\n")


if __name__ == "__main__":
    get_motif_logo(output_folder_path)
    array_PWM_with_pseudo_counts = data_handle.create_frequency_matrix_with_pseudo_counts(PWM_path, True)
    pretty_print_of_PWM(array_PWM_with_pseudo_counts)

