import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
import create_data_for_Homer
sys.path.insert(0, base_path+'/CNN/')
import test_CNN


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'run_Homer_find_denovo_motifs.py')


    for species_name in project.species:
        print("species_name = ", species_name, " : ")
        if species_name != "simulated":
            # species_dir = os.path.join(project.text_samples_base_dir, species_name)
            Homer_dir = create_data_for_Homer.create_Homer_directory(project, species_name)
        else:
            # species_dir = project.text_samples_base_dir
            Homer_dir = create_data_for_Homer.create_Homer_directory(project)

        # run Homer findMotifs:
        script = "~tommy/Work/HOMER/bin/findMotifs.pl"
        positive_samples_path = os.path.join(Homer_dir, species_name + "_positive.fa")
        negative_samples_path = os.path.join(Homer_dir, species_name + "_negative.fa")
        output_directory = os.path.join(Homer_dir, "motifResults/")
        homer_results_file = os.path.join(Homer_dir, "homer_results.txt")
        os.system(script + " " + positive_samples_path + " fasta " + output_directory +
                  " -fasta " + negative_samples_path +
                  " -S 5 -p 16 -bits -mset vertebrates 2>&1 | tee " + homer_results_file)
    print("End!")


if __name__ == "__main__":
    main()
