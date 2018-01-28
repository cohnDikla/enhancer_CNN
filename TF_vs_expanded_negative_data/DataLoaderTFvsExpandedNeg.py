import os
import sys
# get the directory of the script being run:
base_path = os.path.dirname(os.path.abspath(__file__))
base_path = base_path[:-len("TF_vs_expanded_negative_data")]
sys.path.insert(0, base_path+'/TF_vs_negative_data')
from DataLoaderTFvsNeg import DataLoaderTFvsNeg
sys.path.insert(0, base_path+'/CNN')


class DataLoaderTFvsExpandedNeg(DataLoaderTFvsNeg):
    """
    Data Loader class for the project of TF_vs_expanded_negative_data.
    """
    species_names_map = {"Canis_familiaris": "Dog", "Homo_sapiens": "Human",
                                "Monodelphis_domestica": "Opossum", "Mus_musculus": "Mouse"}

    def __init__(self, project):
        # self.ratio_of_samples_from_all_species = 0.6
        # self.ratio_of_samples_from_all_species = 0.8
        self.ratio_of_samples_from_all_species = 1.0
        super(DataLoaderTFvsExpandedNeg, self).__init__(project)
        self.ratio_of_samples_from_all_species = 1.0
        self.maximal_k = 4
        # self.num_times_negative_data_is_taken = 2

    def get_all_positive_and_negative_samples(self, num_times_negative_data_is_taken):
        original_negative_biological_samples_dir = os.path.join(base_path, "H3K27ac_vs_k_shuffle", "data", "samples")
        original_positive_and_k_shuffle_samples_dir = os.path.join(base_path, "TF_vs_k_shuffle", "data", "samples")
        # positive data:
        all_species_positive_files = []
        for species_name in self.project.species:
            if "All_species" in species_name:
                continue
            positive_samples_files_one_species = []
            for k in range(self.maximal_k):
                species_dir = os.path.join(original_positive_and_k_shuffle_samples_dir, species_name)
                samples_path = os.path.join(species_dir, "positive_samples")
                positive_samples_files_one_species.append(samples_path)
            all_species_positive_files.append(positive_samples_files_one_species)

        # negative data:
        negative_samples_files_biological_and_all_k = []
        for species_name in self.project.species:
            if "All_species" in species_name:
                continue
            negative_samples_files_biological_and_all_k_one_species = []
            new_species_name = self.species_names_map[species_name]
            # biological negative data:
            species_dir = os.path.join(original_negative_biological_samples_dir, new_species_name)
            samples_path = os.path.join(species_dir, new_species_name + "_new_negative_samples.txt")
            new_samples_path = os.path.join(species_dir, new_species_name + "_new_negative_samples_"
                                            + str(num_times_negative_data_is_taken)+"_times"+".txt")
            with open(new_samples_path, 'w') as out_new:
                for n in range(num_times_negative_data_is_taken):
                    with open(samples_path) as original_samples_file:
                        for line in original_samples_file:
                            if "\n" not in line:
                                new_line = line+"\n"
                            else:
                                new_line = line
                            out_new.write(new_line)
            negative_samples_files_biological_and_all_k_one_species.append(new_samples_path)
            # k-shuffle negative data:
            for k in range(self.maximal_k):
                # negative_samples_files_one_k = []
                species_dir = os.path.join(original_positive_and_k_shuffle_samples_dir, species_name)
                species_k_dir = os.path.join(species_dir, self.project.k_let_dirs[k])
                samples_path = os.path.join(species_k_dir, "negative_samples")
                negative_samples_files_biological_and_all_k_one_species.append(samples_path)

            negative_samples_files_biological_and_all_k.append(negative_samples_files_biological_and_all_k_one_species)


        self.positive_samples_files = all_species_positive_files
        self.negative_samples_files = negative_samples_files_biological_and_all_k
        self.num_times_negative_data_is_taken = num_times_negative_data_is_taken
        print("len(self.positive_samples_files) = ", len(self.positive_samples_files))
        print("len(self.negative_samples_files) = ", len(self.negative_samples_files))



