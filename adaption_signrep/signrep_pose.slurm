#!/bin/bash
#SBATCH --job-name=pose_l1
#SBATCH --partition=128cpu_256mem
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/pose_l1_%A_%a.out

export MEDIAPIPE_DISABLE_GPU=1
source ~/.bashrc
conda activate vjepa2-312

python signrep_pose_l1.py \
    --video-dir /hpchome/baptiste26/data/RUV/L1_signers/cropped \
    --out-dir /hpchome/baptiste26/data/RUV/L1_signers/pose \
    --task-id $SLURM_ARRAY_TASK_ID \
    --num-tasks 50