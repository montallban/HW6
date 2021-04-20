#!/bin/bash

#SBATCH --partition=normal
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=10000
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results_hw6/GRU_%04a.txt
#SBATCH --error=results_hw6/error_GRU_%04a.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=GRU
#SBATCH --mail-user=michael.montalbano@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/mcmontalbano/HW6
#SBATCH --array=0-5
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
source ~fagg/pythonenv/tensorflow/bin/activate

python base.py -exp_index $SLURM_ARRAY_TASK_ID -epochs 1000 -experiment_type 'test' -network 'recurrent'
