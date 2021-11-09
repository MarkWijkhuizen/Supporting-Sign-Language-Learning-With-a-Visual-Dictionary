import argparse
import psutil

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")

parser.add_argument('--dataset', type=str, required=True, help='Dataset Used') # choices=['jester', 'nvgesture', 'chalearn']
parser.add_argument('--modality', type=str, default='RGBFlow') # choices=['RGB', 'Flow', 'RGBDiff', 'RGBFlow']
parser.add_argument('--train_list', type=str,default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default='BNInception')
parser.add_argument('--num_segments', type=int, default=4)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--num_motion_subset', type=int, default=None, help='subset of motion frames to use')
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.50, type=float, metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--dropout_extra', '--do-extra', default=0.00, type=float, metavar='DO EXTRA', help='extra dropout layer in consensus module')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', type=float, nargs="+", metavar='LRSteps', default=[25, 40], help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--lr_decay', '--lrd', default=0.10, type=float, help='learning rate decay')
parser.add_argument('--clip_gradient', '--gd', default=20, type=float, metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")


# ========================= Monitor Configs ==========================
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval_freq', '-ef', default=1, type=int, metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', nargs=2, type=int, default=[0, 0], help='number of worker for training and validation dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str2bool, help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output',type=str, default='output')

# ========================= Custom Stuff =============================
parser.add_argument('--debug',type=bool, default=False)
parser.add_argument('--description',type=str, required=True, help='Description of Training RUn')
parser.add_argument('--optimizer',type=str, default='SGD')
parser.add_argument('--input_size',type=int, default=224)
parser.add_argument('--dataset_subset',type=int, nargs=2, default=(None, None), help='Train and Validation subset amount')
parser.add_argument('--fps', type=int, help='FPS of Dataset Used')
parser.add_argument('--validate', type=str, default=None, help='Is there a Validation Set')
parser.add_argument('--amp', type=str2bool, default=False, help='Use Automatic Mixed Precision')
parser.add_argument('--dataset_ratio', type=float, default=100, help='Percentage of train dataset to use')
parser.add_argument('--compression', type=str, default=None, help='Compression type of MFF\'s', choices=[None, 'bz2', 'lzma'])
parser.add_argument('--num_class', type=int, required=True, help='Number of classes')
parser.add_argument('--cache_val', type=str2bool, default=False, help='cache validation MFF\'s')
parser.add_argument('--sim_label', type=str2bool, default=False, help='use similarity labels for NGT dataset')
parser.add_argument('--sim_label_eps', type=float, default=0.0, help='minimum similarity value')
parser.add_argument('--sim_label_exp', type=float, default=1.0, help='similarity exponent')
parser.add_argument('--sim_label_target_scale', type=float, default=1.0, help='similarity target label scaling')
parser.add_argument('--multi_sim_label', type=str2bool, default=False, help='use per aspect similarity label')
parser.add_argument('--person_label', type=str2bool, default=False, help='use person labels for NGT dataset')
parser.add_argument('--frame_range', type=int, required=True, help='Frame range of the MFF\'s')
parser.add_argument('--ndcg', type=str2bool, default=False, help='display normalized discounted cummulative gain validation metric')
parser.add_argument('--ndcg_detailed', type=str2bool, default=False, help='display normalized discounted cummulative gain validation metric for each part')
parser.add_argument('--confusion_matrix', type=str2bool, default=False, help='confusion matrix')
parser.add_argument('--index_label', type=str2bool, default=False, help='use index of sample as label (used for subsets)')
parser.add_argument('--of_kfe', type=str2bool, default=False, help='use optical flow key frame extraction datasets')
parser.add_argument('--op_kfe', type=str2bool, default=False, help='use open pose key frame extraction datasets')
parser.add_argument('--train_augmentations', type=str2bool, default=False, help='use train augmentations crop/pad, rotate and gridmask')
parser.add_argument('--loc_cat', type=str2bool, default=False, help='use location categorie metric')
parser.add_argument('--mov_cat', type=str2bool, default=False, help='use movement categorie metric')
parser.add_argument('--best_k', type=int, required=True, help='value k for best epoch precision@k criterion')
parser.add_argument('--save_best_weights_only', type=str2bool, default=False, help='saves only the best weights, not each epoch')
parser.add_argument('--silence_tqdm', type=str2bool, default=True, help='silence tqdm output in sbatch job')
parser.add_argument('--gridmask', type=str2bool, default=False, help='use GridMask data augmentation')
parser.add_argument('--create_embeddings', type=str2bool, default=False, help='Create MFF\'s Embeddings')
parser.add_argument('--label_smoothing', type=float, default=None, help='Label smoothing for labels')
parser.add_argument('--arcface_head', type=str2bool, default=False, help='Label smoothing for labels')
parser.add_argument('--arcface_s', type=float, default=1.0, help='ArcFace logits scale')
parser.add_argument('--arcface_m', type=float, default=0.0, help='ArcFace margin in radians')
parser.add_argument('--arcface_ls', type=float, default=0.0, help='ArcFace label smoothing')
parser.add_argument('--save_ndcg', type=str2bool, default=False, help='evaluation ndcg scores')
