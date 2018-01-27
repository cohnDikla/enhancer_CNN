import os
from CNN_structure import CNN_structure

H3K27ac_species_names_ordered = ["Human",
                         "Dog",
                         "Dolphin",
                         "Tasmanian_Devil",
                         "Ferret",
                         "Guinea_pig",
                         "Tree_shrew",
                         "Marmoset",
                         "Cat",
                         "Cow",
                         "Opossum",
                         "Mouse",
                         "Macaque",
                         "Rabbit",
                         "Naked_mole_rat",
                         "Rat",
                         "Pig",
                         "All_species_238000",
                         "All_species_14000"]


TF_species_names_ordered = ["Canis_familiaris", "Gallus_gallus", "Homo_sapiens",
                            "Monodelphis_domestica", "Mus_musculus",
                            "All_species_60000", "All_species_12000"]

TF_species_names_ordered_negative_data = ["Canis_familiaris", "Homo_sapiens",
                                          "Monodelphis_domestica", "Mus_musculus",
                                          "All_species_60000", "All_species_12000"]


simulated_data_dirs = ["normal_dist_centers", "regular_centers"]


class Project:
    MAXIMAL_K = 9
    k_let_dirs = ["preserving_"+str(k)+"-let_counts/" for k in range(1, MAXIMAL_K+1)]

    def __init__(self, project_name_and_PWM, base_path, k=None, normal_distribution=False, sigma=None, num_times_negative_data_is_taken=None):
        # check if receive specific PWM
        if "_CEBPA_JASPAR" in project_name_and_PWM or "_HNF4A_JASPAR" in project_name_and_PWM\
                or "_denovo" in project_name_and_PWM:
            motif_name = "_".join(project_name_and_PWM.split("_")[2:])
            self.project_name = project_name_and_PWM[:-len(motif_name)-1]
            self.PWM = project_name_and_PWM[len(self.project_name) + 1:]
        else:
            self.project_name = project_name_and_PWM
            self.PWM = None
        print("self.project_name = ", self.project_name)
        print("self.PWM = ", self.PWM)
        self.all_projects_base_bath = base_path
        self.project_base_path = os.path.join(base_path, self.project_name)
        self.k = k
        self.num_times_negative_data_is_taken = num_times_negative_data_is_taken
        self.board_folder = os.path.join(self.project_base_path, "board/")
        self.basic_output_dir = os.path.join(self.project_base_path, "output")
        self.CNN_output_dir = os.path.join(self.basic_output_dir, 'CNN')
        self.PSSM_output_dir = os.path.join(self.basic_output_dir,
                                            'PSSM_straw_man_model')
        self.SVM_output_dir = os.path.join(self.basic_output_dir, 'SVM')
        self.number_of_samples = self.get_number_of_samples()
        self.normal_distribution = normal_distribution
        self.sigma = sigma
        self.base_dir_data_path = os.path.join(self.project_base_path, "data")
        self.original_samples_dir = None
        self.checkpoints_folder_tmp = os.path.join(self.project_base_path, "checkpoints_tmp")

        if self.project_name == "simulated_data":
            self.distribution_samples_center_dir = simulated_data_dirs[0] if self.normal_distribution else simulated_data_dirs[1]
            self.checkpoints_folder = os.path.join(self.project_base_path, "checkpoints/",
                                                   self.distribution_samples_center_dir)
            data_sigma_dir = os.path.join(self.project_base_path,
                                                 "data", self.distribution_samples_center_dir,
                                                      self.PWM, "sigma_"+str(self.sigma))
            if not os.path.exists(data_sigma_dir) and not os.path.isdir(data_sigma_dir):
                print("make directory: ", data_sigma_dir)
                os.makedirs(data_sigma_dir)
            self.data_sigma_dir = data_sigma_dir

            self.text_samples_base_dir = os.path.join(data_sigma_dir, "samples")
            self.svm_samples_base_dir = os.path.join(data_sigma_dir, "svm_samples")
            self.samples_base_dir = os.path.join(data_sigma_dir, "npy_files")

            self.species = ["simulated"]
            self.output_results_file = os.path.join(self.CNN_output_dir,
                                                    self.distribution_samples_center_dir,
                                                     self.PWM,
                                                    'CNN_train_models_summary_'
                                                    + self.distribution_samples_center_dir + '.txt')
            self.test_file = os.path.join(self.CNN_output_dir, self.distribution_samples_center_dir,
                                                     self.PWM,
                                              'CNN_test_output_' + self.distribution_samples_center_dir + '.txt')

            self.base_dir_test_CNN_results = os.path.join(self.CNN_output_dir, self.distribution_samples_center_dir,
                                                      self.PWM)
            if self.PWM != "denovo":
                self.CNN_structure = CNN_structure(self.project_name + "_" + self.distribution_samples_center_dir)
        else:
            self.base_dir_test_CNN_results = self.CNN_output_dir
            self.svm_samples_base_dir = os.path.join(self.base_dir_data_path, "svm_samples")
            self.CNN_structure = CNN_structure(self.project_name)
            self.samples_base_dir = os.path.join(self.base_dir_data_path, "npy_files")
            self.text_samples_base_dir = os.path.join(self.base_dir_data_path, "samples")
            if self.project_name=="negative_data_vs_k_shuffle":
                self.species = H3K27ac_species_names_ordered[:-2]
            if self.project_name.startswith("H3K27ac"):
                self.species = H3K27ac_species_names_ordered
                if "expanded" in self.project_name:
                    self.text_samples_base_dir = os.path.join(self.base_dir_data_path,
                                                              str(self.num_times_negative_data_is_taken) + "_times_negative_data",
                                                              "samples")
                    self.samples_base_dir = os.path.join(self.base_dir_data_path,
                                                         str(self.num_times_negative_data_is_taken) + "_times_negative_data",
                                                         "npy_files")

            elif self.project_name.startswith("TF"):
                if "k_shuffle" in self.project_name:
                    self.species = TF_species_names_ordered
                elif "negative_data" in self.project_name:
                    self.species = TF_species_names_ordered_negative_data
                    if "expanded" in self.project_name:
                        self.text_samples_base_dir = os.path.join(self.base_dir_data_path,
                                                                  str(self.num_times_negative_data_is_taken) + "_times_negative_data",
                                                                  "samples")
                        self.samples_base_dir = os.path.join(self.base_dir_data_path,
                                                                  str(self.num_times_negative_data_is_taken) + "_times_negative_data",
                                                                  "npy_files")


            if self.k is None:
                self.checkpoints_folder = os.path.join(self.project_base_path, "checkpoints/")

                self.output_results_file = os.path.join(self.CNN_output_dir,
                                                        'CNN_models_summary.txt')
                self.test_file = os.path.join(self.CNN_output_dir, 'CNN_test_output.txt')
            else:
                self.checkpoints_folder = os.path.join(self.project_base_path,
                                                       "checkpoints",
                                                        self.k_let_dirs[k-1])
                self.output_results_file = os.path.join(self.CNN_output_dir,
                                                    "CNN_models_summary_k_" + str(k) + ".txt")
                self.test_file = os.path.join(self.CNN_output_dir,
                                              "CNN_test_output_k_" + str(k) + ".txt")

    def print_project_details(self, out_file):
        with open(out_file, 'a') as f:  # appends the line at the end of the file
            f.write('num conv layers: ' + str(self.CNN_structure.num_conv_layers) + '\t')
            for layer_num in range(1, self.CNN_structure.num_conv_layers+1):
                kernel_shape, num_kernels = self.CNN_structure.get_kernels_shape_and_number(layer_num)
                f.write('layer'+str(layer_num)+'_num_kernels: '+str(num_kernels)+'\t'
                        'layer'+str(layer_num)+'_filter_size: '+str(kernel_shape)+'\t')
            f.write('layer_affine_1_size: '+str(self.CNN_structure.affine1_size)+'\t')
            f.write('layer_affine_2_size: '+str(self.CNN_structure.affine2_size)+'\t')
            f.write('dropout_prob: ' + str(self.CNN_structure.dropout_prob) + '\t')
            f.write('mini_batch_size: ' + str(self.CNN_structure.mini_batch_size) + '\t')
            f.write('max_pool_shapes: '+str(self.CNN_structure.max_pool_shapes) + '\t')
            if self.project_name == "simulated_data":
                if self.normal_distribution:
                    f.write('distribution of sample centers is normal' + '\t')
                else:
                    f.write('distribution of sample centers is regular' + '\t')
            f.write('\n\n\n')

    def get_number_of_samples(self):
        # same number for positive and negative samples
        #  total number of samples is 2*NUMBER_OF_SAMPLES
        if self.project_name == "simulated_data":
            return 10000
        elif self.project_name == "TF_vs_negative_data":
            return 12000
        elif self.project_name == "TF_vs_k_shuffle":
            return 12000
        elif self.project_name == "H3K27ac_vs_negative_data":
            return 14000
        elif self.project_name == "H3K27ac_vs_k_shuffle":
            return 14000
        elif self.project_name == "negative_data_vs_k_shuffle":
            return 14000
        elif self.project_name == "TF_vs_expanded_negative_data":
            return 12000*5
        elif self.project_name == "H3K27ac_vs_expanded_negative_data":
            return 14000*5





