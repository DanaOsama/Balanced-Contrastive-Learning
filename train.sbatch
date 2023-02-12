#!/bin/bash
#SBATCH --job-name=isic
#SBATCH --output=slurm-logs/output-%A-%a.txt 
#SBATCH --partition=ai702
#SBATCH --reservation=ai702
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

python main.py --lr 0.05 -p 200 --epochs 400  \
                --arch resnext50 --use_norm True \
                --wd 1e-4 --cos True --cl_views sim-sim \
                --workers 16 --batch-size 110 --schedule [360, 380] \
                --alpha 2.0 --beta 0.6