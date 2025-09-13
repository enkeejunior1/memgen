#!/bin/bash
#SBATCH --job-name=toy_1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0
# #SBATCH --mem=32G
# #SBATCH --time=24:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate trak
cd $SLURM_SUBMIT_DIR

# Actual work
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

python3 train_mup_sweep.py