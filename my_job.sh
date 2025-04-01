#!/bin/bash
#SBATCH --partition=main          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=BRIDGE         # Assign a short name to your job
#SBATCH --constraint=hal --nodes=1                 # Number of nodes you require
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --mem=16000               # Real memory (RAM) required (MB)
#SBATCH --time=10:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm.%N.%j.out  # STDOUT output file
#SBATCH --error=slurm.%N.%j.err   # STDERR output file (optional)

# Load conda environment properly
source /home/jc3585/anaconda3/etc/profile.d/conda.sh
conda activate ML
echo "($CONDA_DEFAULT_ENV)"
echo "$(which python)"
srun python3 main.py