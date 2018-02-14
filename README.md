Enhancer Identification from DNA sequence using Transfer and Adversarial Deep Learning
======================================================================================

General:
---------

This package implements deep-learning based training algorithms for enhancer classification, described in [1]

Authors: Dikla Cohn, Or Zuk and Tommy Kaplan

Prerequisites:
--------------
Use:
Python 3.5.2+ and Python2.7
tensorflow 1.1.0
used on Linux machine, gpu (nvidia tesla M60)


For k-shuffling we used the uShuffle tool [2]:
link for downloading the uShuffle library: 
http://digital.cs.usu.edu/~mjiang/ushuffle/
How to Use uShuffle in Python:
http://digital.cs.usu.edu/~mjiang/ushuffle/python.html
Build a shared library ushuffle.so and save it in the main directory (enhancer_CNN/). 

For finding denovo motifs and comparing to known motifs we used the Homer tool [3].



Data Preparation:
-----------------
For all projects (except the simulated_data project):
save positive (and negative if needed) samples as text files to: /<project_name>/data/samples/<species_name>/  <br /> 
The files should contain one line for each sample string (of A,C,G,T only, no N's)  <br /> 
Name the files: 
negative_samples
positive_samples
(negative samples not required for k_shuffle projects).
Each file should contain:  <br /> 
for TF data - 12K lines (samples)
for H3K27ac data - 14K lines (samples)
- as in the given example files, located in: TF_vs_negative_data/data/samples/example/
you can use the: create_species_dirs.py script to create species directories (already created in each project). 

 
Specifically for the simulated_data project, no need to create positive and negative samples in advance.
Run:  <br /> 
python2.7 /simulated_data/run_data_loader.py simulated_data_<motif_name> normal_<sigma>  <br /> 
For example:  <br /> 
python2.7 /simulated_data/run_data_loader.py simulated_data_CEBPA_JASPAR normal_40  <br /> 
This module creates the simulated data of one TF: CEBBA or HNF4A.
Each sample contains a short sequence sampled from the PWM of the TF.  <br /> 
The location of the planted motif is sampled with normal distribution around the center of each sample, according to the given <sigma> value. (We used sigma=40).  <br /> 
This module also writes all created data samples and labels both as text files and as numpy binary files to: 
/simulated_data/data/normal_dist_centers/<motif_name>/samples/  <br /> 
/simulated_data/data/normal_dist_centers/<motif_name>/npy_files/  <br /> 
The generated files contain 10K samples in positive samples and 10K samples in negative samples.


Run:
----

1. data loader: (from string sequences (ACGT...) to npy files) - run with a specific project dir: 

for simulated data project:
python2.7 simulated_data/run_data_loader.py simulated_data_CEBPA_JASPAR normal_<sigma>

for other projects:
python2.7 /<project_name>/run_data_loader.py [<k>]

for the k-shuffle projects (TF_vs_k_shuffle, H3K27ac_vs_k_shuffle or negative_data_vs_k_shuffle), before running the above run_data_loader.py, run:
python2.7 /<project_name>/data_loader_<project_name>.py

for example:
python2.7 /TF_vs_k_shuffle/data_loader_TF_vs_k_shuffle.py
python2.7 /negative_data_vs_k_shuffle/data_loader_negative_data_vs_k_shuffle_each_species.py
This module creates data for each species separately, and for all values of k (k=1,...,9).

2. CNN train:
CNN_trainer creates tar files of new network models, and saves them to <project_name>/checkpoints dir. 

for simulated_data project:
python3 /CNN/run_test_CNN.py simulated_data_CEBPA_JASPAR <num_runs> <num_epochs> normal_<sigma>
for example:
python3 /CNN/CNN_trainer.py simulated_data_CEBPA_JASPAR 50 20 normal_40

for other projects:
python3 /CNN/CNN_trainer.py <project_name> <num_runs> <num_epochs> [<k>]
for example:
python3 /CNN/CNN_trainer.py TF_vs_negative_data 50 20
python3 /CNN/CNN_trainer.py TF_vs_k_shuffle 50 20 4
python3 /CNN/CNN_trainer.py H3K27ac_vs_negative_data 50 20
python3 /CNN/CNN_trainer.py H3K27ac_vs_k_shuffle 50 20 4

for training on all k values:
sbatch /<project_name>/train_all_k.sh

Before testing:
copy tar files from: <project_name>/checkpoints dir to: <project_name>/checkpoints_tmp dir,
such that checkpoints_tmp dir will contain only tar files of models you wish to test on.

3. CNN test:
for simulated_data project:
python3 /CNN/run_test_CNN.py simulated_data_CEBPA_JASPAR normal_<sigma>
for example:
python3 /CNN/run_test_CNN.py simulated_data_CEBPA_JASPAR normal_40

for other projects:
python3 /CNN/run_test_CNN.py <project_name> [<k>]
for testing networks trained on all k values:
sbatch /<project_name>/test_all_k.sh

4. show convolution:
for simulated_data project:
python3 /CNN/show_convolution.py simulated_data_CEBPA_JASPAR <normal_sigma>
for other projects:
python3 /CNN/show_convolution.py <project_name> [<k>]

5. tensor visualization: (used for Figure4 and Figure6)
for simulated_data project:
python3 /CNN/tensor_visualization.py simulated_data_CEBPA_JASPAR <normal_sigma>
for other projects:
python3 /CNN/tensor_visualization.py <project_name> [<k>]

6. Compare to known motifs (using the Homer tool - compareMotifs.pl):  (used for Figure 4 and Figure 6)
for simulated_data project:
python3 /motifs/read_filters_and_run_Homer_compare_motifs.py simulated_data_CEBPA_JASPAR <normal_sigma>
for other projects:
python3 /motifs/read_filters_and_run_Homer_compare_motifs.py <project_name> [<k>]

7. Homer find denovo and known motifs (using the Homer tool - findMotifs.pl):
for simulated_data project:
python3 /create_data_for_Homer.py simulated_data_CEBPA_JASPAR <normal_sigma>
python3 /run_Homer_find_denovo_motifs.py simulated_data_CEBPA_JASPAR <normal_sigma> 

for other projects:
python3 /create_data_for_Homer.py <project_name> [<k>]
python3 /run_Homer_find_denovo_motifs.py <project_name> [<k>]



PSSM straw man model: (both with and without prior knowlegde regarding the distribution of planted motif's location)
First PSSM model - uses PWM of CEBPA transcription factor from JASPAR:
for simulated_data project:
python3 /PSSM_straw_man_model/straw_man_model.py simulated_data_CEBPA_JASPAR normal_<sigma>
for other projects:
python3 /PSSM_straw_man_model/straw_man_model.py <project_name>

Second PSSM model - uses PWM of denovo motif (first result in Homer findMotifs, when running on positive vs. negative data):
for simulated_data project:
python3 /PSSM_straw_man_model/straw_man_model.py simulated_data_denovo normal_<sigma>
for other projects:
python3 /PSSM_straw_man_model/straw_man_model.py <project_name>

for example,
python3 /PSSM_straw_man_model/straw_man_model.py simulated_data_CEBPA_JASPAR normal_40
python3 /PSSM_straw_man_model/straw_man_model.py simulated_data_denovo normal_40




Figures:
--------

The code shown below was used to generate the figures in the paper [1] 

Figure2:
python3 /roc_comparison.py simulated_data_CEBPA_JASPAR normal_<sigma>
for example,
python3 /roc_comparison.py simulated_data_CEBPA_JASPAR normal_40

Figure3 and supp.Figure3:
For TF projects:
python3 /CNN/display_heatmap_TF.py <project_name> [<k>]

for example:
python3 /CNN/display_heatmap_TF.py TF_vs_negative_data
python3 /CNN/display_heatmap_TF.py TF_vs_k_shuffle <k>
and similarly for enhancer projects:
python3 /CNN/display_heatmap_enhancer.py H3K27ac_vs_negative_data
python3 /CNN/display_heatmap_enhancer.py H3K27ac_vs_k_shuffle <k>

Figure5:
python3 /CNN/display_k_graph_different_models.py TF_vs_k_shuffle negative_data_vs_k_shuffle H3K27ac_vs_k_shuffle


Acknowledgment
--------------
This package was developed by Dikla Cohn, as part of work on the paper [1]. Please cite this paper if using the package


References
----------
[1] 
["Enhancer Identification using Transfer and Adversarial Deep Learning of DNA Sequences"](https://www.biorxiv.org/content/early/2018/02/13/264200)
D. Cohn, O. Zuk and T. Kaplan (Biorxiv, 2018)

[2] uShuffle: A useful tool for shuffling biological sequences while preserving the k-let counts.
Jiang, M. et al., 2008
BMC Bioinformatics 2008 9:192

[3] Simple Combinations of Lineage-Determining Transcription Factors Prime cis-Regulatory Elements Required for Macrophage and B Cell Identities.
Heinz, S. et al., 2010
MolCell,38(4), 576-589.
doi:10.1016/j.molcel.2010.05.004






