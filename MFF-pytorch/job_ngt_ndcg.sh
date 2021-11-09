#!/bin/bash
#SBATCH --partition=medium
#SBATCH --error error.log
#SBATCH -w mlp13
#SBATCH -N 1 -n 12
#SBATCH --mem=124

source ../bin/activate

python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts2_1 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 2.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar

python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts2_2 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 2.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar

python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts2_3 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 2.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar


python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts4_1 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 4.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar

python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts4_2 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 4.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar

python main.py --description ngt_300_l2r_500_epochs_fr3_eps001_exp1_ts4_3 --op_kfe True --sim_label_eps 0.01 --sim_label_exp 1.0 --sim_label_target_scale 4.0 --dataset ngt_300 --fps 10 --epochs 500 --arch efficientnetv2_s --lr_steps -1 -1 --validate val --optimizer SGD --consensus_type MLP --compression lzma --workers 10 6 --batch_size 30 --dropout 0.50 --dropout_extra 0.25 --num_class 300 --optimizer SGD --amp 0 --lr_decay 0.25 --cache_val 1 --person_label 0 --frame_range 3 --sim_label 1 --train_augmentations 1 --best_k 20 --save_best_weights_only True --ndcg 1 --ndcg_detailed 0 --resume model/MFF_jester_RGBFlow_efficientnetv2_segment4_3f1c_jester_4mff_100p_no_loss_scale_no_gridmask_best.pth.tar

