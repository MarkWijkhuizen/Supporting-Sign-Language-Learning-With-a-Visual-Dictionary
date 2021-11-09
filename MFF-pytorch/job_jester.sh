#!/bin/bash
#SBATCH --partition=medium
#SBATCH -w mlp13
#SBATCH -N 1 -n 24
#SBATCH --mem=124GB

source ../bin/activate

nice -n 19 python main.py --description jester_efficientnetv2b0 --silence_tqdm True --dataset jester --epochs 45 --arch tf_efficientnetv2_b0 --lr_steps 25 40 --validate val --optimizer SGD --amp 0 --consensus_type MLP --num_segments 4 --num_motion 3 --num_class 27 --workers 16 4 --train_augmentations 1 --num_class 27 --frame_range 2 --of_kfe 1 --best_k 1 --dropout 0.50
