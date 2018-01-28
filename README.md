
README.md - running code for the paper:

Enhancer Identification from DNA sequence using Transfer and Adversarial Deep Learning

Dikla Cohn, Or Zuk and Tommy Kaplan.


use Python 3.5.2+
use tensorflow 1.1.0
used on Linux machine, gpu (nvidia tesla M60)



1. data loader: (from string sequences (ACGT...) to npy files) - run from a specific project dir: 
python2.7 /simulated_data/run_data_loader.py

2. CNN train:
python3 /CNN/CNN_trainer.py <project_name> <num_runs> <num_epochs> [<k=None> or <normal_sigma>]
all k's:
sbatch /TF_vs_k_shuffle/train_all_k.sh

3. CNN test:
python3 /CNN/run_test_CNN.py <project_name> [<k=None> or <normal_sigma>]

4. show_convolution:
python3 /CNN/show_convolution.py <project_name> [<k=None> or <normal_sigma>]

5. tensor_visualization:
python3 /CNN/tensor_visualization.py <project_name> [<k=None> or <normal_sigma>]

6. Compare motifs (using the Homer tool - compareMotifs.pl):
python3 /motifs/read_filters_and_run_Homer_compare_motifs.py <project_name> [<k=None> or <normal_sigma>]


Homer find denovo and known motifs (using the Homer tool):
python3 /create_data_for_Homer.py <project_name> [<k=None> or <normal_sigma>]
python3 /run_Homer_find_denovo_motifs.py <project_name> [<k=None> or <normal_sigma>]
