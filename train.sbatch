#!/bin/bash
#SBATCH --job-name=hellompi
#SBATCH --output=/nfs/users/ext_group6/Salwa_Project/slurm-logs/output-%A-%a.txt 
#SBATCH --partition=ai702
#SBATCH --reservation=ai702
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

python main.py --lr 0.1 -p 200 --epochs 90 --arch resnet50 --use_norm True --wd 5e-4 --cos True --cl_views sim-sim --workers 16 --batch-size 256