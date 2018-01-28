import sys
import os
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path+"/CNN/")
import test_CNN


def create_Homer_directory(project, species_name=None):
    Homer_dir = os.path.join(project.text_samples_base_dir, "Homer_denovo_motifs")
    if species_name:
        Homer_dir = os.path.join(Homer_dir, species_name)
    if not os.path.exists(Homer_dir) and not os.path.isdir(Homer_dir):
        print("make directory: ", Homer_dir)
        os.makedirs(Homer_dir)
    return Homer_dir


def write_data_file_for_Homer(is_positive, input_file_path, output_file_path):
    with open(output_file_path, 'w+') as out_file:
        with open(input_file_path) as samples:
            line_counter = 1
            if is_positive:
                for line in samples:
                    out_file.write(">pos_" + str(line_counter) + "\n")
                    out_file.write(line)
                    line_counter += 1
            else:
                for line in samples:
                    out_file.write(">neg_"+str(line_counter)+"\n")
                    out_file.write(line)
                    line_counter += 1
    print("finish writing file : ", out_file)


def main():
    project = test_CNN.get_project_and_check_arguments(sys.argv, 'create_data_for_Homer.py')
    for species_name in project.species:
        print("species_name = ", species_name, " : ")
        if species_name != "simulated":
            species_dir = os.path.join(project.text_samples_base_dir, species_name)
            Homer_dir = create_Homer_directory(project, species_name)
        else:
            species_dir = project.text_samples_base_dir
            Homer_dir = create_Homer_directory(project)
        out_positive_samples_path = os.path.join(Homer_dir, species_name + "_positive.fa")
        out_negative_samples_path = os.path.join(Homer_dir, species_name + "_negative.fa")
        input_positive_samples_path = os.path.join(species_dir, "positive_samples")
        if project.k:
            input_negative_samples_path = os.path.join(species_dir,
                                                       project.k_let_dirs[project.k - 1],
                                                       "negative_samples")
        else:
            input_negative_samples_path = os.path.join(species_dir, "negative_samples")

        write_data_file_for_Homer(True, input_positive_samples_path, out_positive_samples_path)
        write_data_file_for_Homer(False, input_negative_samples_path, out_negative_samples_path)

    print("End!")


if __name__ == "__main__":
    main()
