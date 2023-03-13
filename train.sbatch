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
                --arch resnet50 --use_norm True \
                --wd 5e-4 --cos True --cl_views sim-sim  --classes 5\
                --workers 32 --batch-size 110  \
                --alpha 1.0 --beta 0.4 --ce_loss Focal --logit_adjust 'train' \
                --many_shot_thr 600 --low_shot_thr 200 \
                # --resume "/home/salwa.khatib/bcl/Balanced-Contrastive-Learning/log/aptos_resnet50_batchsize_110_epochs_2000_temp_0.07_lr_0.05_sim-sim_alpha_1.0_beta_0.35_schedule_[860, 880]_recalibrate_False_Salwa_ce_loss_Focal_acmwjv/bcl_ckpt.pth.tar"
                