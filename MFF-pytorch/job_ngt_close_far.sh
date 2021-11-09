#!/bin/bash
#SBATCH --partition=medium
#SBATCH --error error.log
#SBATCH -w mlp13
#SBATCH -N 1 -n 24
#SBATCH --mem=124

source ../bin/activate

python main.py --description ngt_close_far_200_epochs_gm_1 --silence_tqdm yes --dataset ngt_full --fps 10 --epochs 200 --arch efficientnet --lr_steps 80 160 --arch efficientnetv2 --optimizer SGD --consensus_type MLP --validate train --compression lzma --workers 12 8 --batch_size 32 --dropout 0.60 --dropout_extra 0.60 --num_class 3846 --optimizer SGD --amp no --lr_decay 0.25 --cache 0 0 --frame_range 2 --op_kfe yes --train_augmentations 1 --save_best_weights_only yes --best_k 1 --gridmask no --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar