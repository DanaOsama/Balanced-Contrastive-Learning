hostname
python main.py --lr 0.05 --epochs 2000 --dataset aptos \
                --arch resnext50 --use_norm True \
                --wd 5e-4 --cos True --cl_views sim-sim  --classes 5 \
                --workers 32 --batch-size 110  \
                --alpha 1.0 --beta 0.6 --ce_loss LC --logit_adjust 'train' \
                --many_shot_thr 600 --low_shot_thr 200 --user_name 'dana'