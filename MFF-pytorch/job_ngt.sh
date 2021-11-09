#!/bin/bash
#SBATCH --partition=medium
#SBATCH --error error.log
#SBATCH -w mlp13
#SBATCH -N 1 -n 24
#SBATCH --mem=124

source ../bin/activate

python main.py --description arcface_s16_m10 --arcface_head yes --arcface_s 16.0 --arcface_m 0.10 --dataset ngt_300 --num_motion_subset 3 --gridmask no --op_kfe no --fps 10 --epochs 200 --arch efficientnetv2_s --lr_steps 100 -1 --validate val --best_k 20 --optimizer SGD --consensus_type MLP --compression lzma --workers 10 10 --batch_size 30 --dropout 0.50 --dropout_extra 0.50 --num_class 300 --optimizer SGD --amp no --lr_decay 0.25 --cache_val 1 --similarity_label 0 --person_label 0 --frame_range 3 --ndcg 1 --ndcg_detailed 0 --loc_cat 0 --mov_cat 0 --train_augmentations 1 --save_best_weights_only yes --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar
python main.py --description arcface_s16_m20 --arcface_head yes --arcface_s 16.0 --arcface_m 0.20 --dataset ngt_300 --num_motion_subset 3 --gridmask no --op_kfe no --fps 10 --epochs 200 --arch efficientnetv2_s --lr_steps 100 -1 --validate val --best_k 20 --optimizer SGD --consensus_type MLP --compression lzma --workers 10 10 --batch_size 30 --dropout 0.50 --dropout_extra 0.50 --num_class 300 --optimizer SGD --amp no --lr_decay 0.25 --cache_val 1 --similarity_label 0 --person_label 0 --frame_range 3 --ndcg 1 --ndcg_detailed 0 --loc_cat 0 --mov_cat 0 --train_augmentations 1 --save_best_weights_only yes --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar
python main.py --description arcface_s16_m30 --arcface_head yes --arcface_s 16.0 --arcface_m 0.30 --dataset ngt_300 --num_motion_subset 3 --gridmask no --op_kfe no --fps 10 --epochs 200 --arch efficientnetv2_s --lr_steps 100 -1 --validate val --best_k 20 --optimizer SGD --consensus_type MLP --compression lzma --workers 10 10 --batch_size 30 --dropout 0.50 --dropout_extra 0.50 --num_class 300 --optimizer SGD --amp no --lr_decay 0.25 --cache_val 1 --similarity_label 0 --person_label 0 --frame_range 3 --ndcg 1 --ndcg_detailed 0 --loc_cat 0 --mov_cat 0 --train_augmentations 1 --save_best_weights_only yes --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar
python main.py --description arcface_s16_m40 --arcface_head yes --arcface_s 16.0 --arcface_m 0.40 --dataset ngt_300 --num_motion_subset 3 --gridmask no --op_kfe no --fps 10 --epochs 200 --arch efficientnetv2_s --lr_steps 100 -1 --validate val --best_k 20 --optimizer SGD --consensus_type MLP --compression lzma --workers 10 10 --batch_size 30 --dropout 0.50 --dropout_extra 0.50 --num_class 300 --optimizer SGD --amp no --lr_decay 0.25 --cache_val 1 --similarity_label 0 --person_label 0 --frame_range 3 --ndcg 1 --ndcg_detailed 0 --loc_cat 0 --mov_cat 0 --train_augmentations 1 --save_best_weights_only yes --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar
python main.py --description arcface_s16_m50 --arcface_head yes --arcface_s 16.0 --arcface_m 0.50 --dataset ngt_300 --num_motion_subset 3 --gridmask no --op_kfe no --fps 10 --epochs 200 --arch efficientnetv2_s --lr_steps 100 -1 --validate val --best_k 20 --optimizer SGD --consensus_type MLP --compression lzma --workers 10 10 --batch_size 30 --dropout 0.50 --dropout_extra 0.50 --num_class 300 --optimizer SGD --amp no --lr_decay 0.25 --cache_val 1 --similarity_label 0 --person_label 0 --frame_range 3 --ndcg 1 --ndcg_detailed 0 --loc_cat 0 --mov_cat 0 --train_augmentations 1 --save_best_weights_only yes --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar