#!/bin/bash

#SBATCH --job-name=IM999
#SBATCH --output=IM999.out
#SBATCH --error=IM999.err
#SBATCH --time=2:00:00
#SBATCH --partition=a4000

ulimit -n 65535
# Go to the repo (use the path you run from)
cd /mnt/beegfs/lchang2/clean_covisnet/Int8-Quantization-CovisNet

# (Optional but common) ensure logs dir exists
mkdir -p logs

# Activate your env the way your cluster expects
source /mnt/beegfs/lchang2/anaconda3/etc/profile.d/conda.sh
conda activate modelopt

# Run your command
python -m evaluation.export_smoothquant_onnx