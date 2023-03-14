hostname
python main.py --lr 0.05 --epochs 1000 --dataset isic \
                --arch crossformer --use_norm True \
                --wd 1e-4 --cos True --cl_views sim-sim \
                --workers 32 --batch-size 110  \
                --alpha 1.0 --beta 0.4 --ce_loss LC --logit_adjust 'train' \
                --many_shot_thr 1000 --low_shot_thr 200\
                # --resume "/home/salwa.khatib/bcl/Balanced-Contrastive-Learning/log/isic_crossformer_batchsize_110_epochs_1000_temp_0.07_lr_0.05_sim-sim_alpha_1.0_beta_0.35_schedule_[860, 880]_recalibrate_False_Salwa_ce_loss_LC_noyyxz/bcl_ckpt.pth.tar"