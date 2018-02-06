#!/bin/sh

#SBATCH --mail-type=end
#SBATCH --mail-user=diklac03@cs.huji.ac.il
#SBATCH --mem=40g
#SBATCH -c8
#SBATCH --time=2-0
#SBATCH --gres=gpu:1

NUMBERS=$(seq 1 9)
for NUM in ${NUMBERS}; do
    srun python3 /cs/cbio/dikla/projects/CNN/CNN_trainer.py H3K27ac_vs_k_shuffle 50 20 $NUM &
done
wait
