hostname
python -u main.py --lr 0.05 --epochs 1000 --dataset isic \
                --arch resnet50 --use_norm True \
                --wd 2e-4 --cos True --cl_views sim-sim \
                --workers 32 --batch-size 110 \
                --alpha 1.0 --beta 0.35 --ce_loss 'LC' --loss_req 'LC' --logit_adjust 'train' \
                --many_shot_thr 1000 --low_shot_thr 200 --user_name 'salwa' --ema_prototypes True  --pretrained True\
                # --resume "/home/salwa.khatib/bcl-current/bcl/Balanced-Contrastive-Learning/log/isic_resnet50_batchsize_110_epochs_1000_temp_0.07_lr_0.05_sim-sim_alpha_1.0_beta_0.35_schedule_[860, 880]_recalibrate-beta0.99_False_salwa_ce_loss_EQLv2_pretrained_True_prototype_ema_False_lkgqgf/bcl_ckpt.pth.tar"
                # --ema_prototypes True  --recalibrate True  --pretrained True --recalibrate True --recalibrate_static True