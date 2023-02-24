#!/bin/bash
#SBATCH --job-name=bcl-new            # Job name
#SBATCH --output=slurm-logs/bcl.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

hostname
python main.py --lr 0.05 --epochs 2000 --dataset aptos \
                --arch resnext50 --use_norm True \
                --wd 1e-4 --cos True --cl_views sim-sim \
                --workers 32 --batch-size 110  \
                --alpha 1.0 --beta 0.4 --ce_loss LC --logit_adjust 'train'
