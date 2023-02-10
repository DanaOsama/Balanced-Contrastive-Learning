#!/bin/bash
#SBATCH --job-name=hellompi
#SBATCH --output=/nfs/users/ext_group6/Dana_Project/Balanced-Contrastive-Learning/slurm-logs/output.%A_%a.txt   
#SBATCH --reservation=ai702
#SBATCH --partition=ai702

hostname
python main.py  --lr 0.1 -p 200 --epochs 90 --arch resnet50 --use_norm True --wd 5e-4 --cos True --cl_views sim-sim   --workers 16 