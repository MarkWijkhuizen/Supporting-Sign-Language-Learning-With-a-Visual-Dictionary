import argparse

# options
parser = argparse.ArgumentParser(description="MFF testing on the full validation set")

parser.add_argument('--dataset', type=str, default='jester') # choices=['jester', 'nvgesture', 'chalearn']
parser.add_argument('--modality', type=str, default='RGBFlow') # choices=['RGB', 'Flow', 'RGBDiff', 'RGBFlow']
parser.add_argument('--weights', type=str, default='pretrained_models/MFF_jester_RGBFlow_BNInception_segment4_3f1c_best.pth.tar')
parser.add_argument('--arch', type=str, default='BNInception')
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=4)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='MLP', choices=['avg', 'MLP'])
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1)
parser.add_argument('--softmax', type=int, default=0)