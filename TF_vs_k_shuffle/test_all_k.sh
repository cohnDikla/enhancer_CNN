#!/bin/sh

#SBATCH --mail-type=end
#SBATCH --mail-user=diklac03@cs.huji.ac.il
#SBATCH --mem=40g
#SBATCH -c8
#SBATCH --time=2-0
#SBATCH --gres=gpu:1

NUMBERS=$(seq 1 9)
for NUM in ${NUMBERS}; do
    srun python3 CNN/run_test_CNN.py TF_vs_k_shuffle ${NUM};
done
wait

