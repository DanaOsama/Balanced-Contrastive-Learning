hostname
python main.py --lr 0.05 --epochs 1000 --dataset aptos \
                --arch resnext50 --use_norm True \
                --wd 3e-4 --cos True --cl_views sim-sim  --classes 5 \
                --workers 32 --batch-size 110  \
                --alpha 1.0 --beta 0.35 --ce_loss 'LC' --loss_req 'LC' --logit_adjust 'train' \
                --many_shot_thr 600 --low_shot_thr 200 --user_name 'salwa' --ema_prototypes True  --pretrained True \
                # --resume '/home/salwa.khatib/bcl/Balanced-Contrastive-Learning/log/aptos_resnet50_batchsize_110_epochs_2000_temp_0.07_lr_0.05_sim-sim_alpha_1.0_beta_0.35_schedule_[860, 880]_recalibrate-beta0.99_False_mai_ce_loss_EQLv2_pretrained_False_prototype_ema_False_rlmypt/bcl_ckpt.pth.tar'
                # --delayed_start True --pretrained True