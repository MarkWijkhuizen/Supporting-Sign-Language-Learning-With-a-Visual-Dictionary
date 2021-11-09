import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from dataset_custom import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import datasets_video
import pdb
from torch.nn import functional as F
from tqdm import tqdm
from apex import amp

torch.cuda.empty_cache()

# options
parser = argparse.ArgumentParser(description="MFF testing on the full validation set")
parser.add_argument('--dataset', type=str, choices=['jester', 'nvgesture', 'chalearn'])
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'RGBFlow'])
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str, default="efficientnetv2")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=4)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=2)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--num_motion', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='MLP', choices=['avg', 'MLP'])
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1)
parser.add_argument('--softmax', type=int, default=0)
parser.add_argument('--fps', type=int, help='FPS of Dataset Used')
parser.add_argument('--subset', type=str, default='val', help='Dataset subset to use', choices=['val', 'test'])
parser.add_argument('--dataset_subset',type=int, help='Subset to use')
parser.add_argument('--compression', type=str, default=None, help='Compression type of MFF\'s', choices=[None, 'bz2', 'lzma'])
parser.add_argument('--print_freq',type=int, default=1, help='Print frequency of moving precisions')

args = parser.parse_args()
args.gpus = list(range(torch.cuda.device_count()))
# MFF Shape
args.n_frames = args.test_segments * (args.num_motion * 2 + 3)
args.mffs_shape = (args.n_frames, args.input_size, args.input_size)

# Global Variables
fps_postfix = f'_{args.fps}fps' if args.fps else ''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res



categories, args.train_list, args.val_list, args.root_path, prefix = datasets_video.return_dataset(args.dataset, args.modality)
print(f'val list: {args.val_list}')
num_class = len(categories)

model = TSN(num_class, args.test_segments if args.consensus_type in ['MLP'] else 1, args.modality,
          base_model=args.arch,
          consensus_type=args.consensus_type,
          img_feature_dim=args.img_feature_dim,
          )

# MODEL
crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std
input_rescale = model.rescale
model_name = model.name

# AMP
model.to('cuda')
model = amp.initialize(model, opt_level="O1")

# Create Model
model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

checkpoint = torch.load(args.weights)
print(f'model epoch {checkpoint["epoch"]} best prec@1: {checkpoint["best_prec1"]:.3f}')

model.module.load_state_dict(checkpoint['state_dict'])
print(f'Sucesfully Loaded Weights')

data_loader = torch.utils.data.DataLoader(
                TSNDataSet(
                    f'{args.dataset}_{args.test_segments}-MFFs-3f1c_of_kfe_v2_fr2{fps_postfix}_{args.subset}',
                    args.mffs_shape,
                    torchvision.transforms.Compose([
                        # Normalize according to ImageNet means and std
                        GroupNormalize(input_rescale, input_mean, input_std, args.n_frames),
                    ]),
                    is_train=False, debug=False, n=args.dataset_subset, index_label=False, compression=args.compression, oversample=args.test_crops,
                ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

model.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    elif args.modality == 'RGBFlow':
        length = 3 + 2 * args.num_motion # 3 rgb channels and 3*2=6 flow channels 
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data, volatile=True)
    pred = model(input_var)
    if args.softmax==1:
        # take the softmax to normalize the output to probability
        pred = F.softmax(rst)

    pred = pred.data.cpu().numpy().copy()

    if args.consensus_type in ['MLP']:
        pred = pred.reshape(-1, 1, num_class)
    else:
        pred = pred.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape((args.test_segments, 1, num_class))

    # Reshape to test crops
    pred = pred.sum(axis=0)
    pred = np.expand_dims(pred, axis=0)
    return i, pred, label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

top1 = AverageMeter()
top5 = AverageMeter()

for i, (data, label) in tqdm(data_gen):
    if i >= max_num:
        break
    _, pred, _ = eval_video((i, data, label))
    output.append(pred)
    cnt_time = time.time() - proc_start_time
    prec1, prec5 = accuracy(torch.from_numpy(np.mean(pred, axis=0)), label, topk=(1, 5))
    top1.update(prec1, 1)
    top5.update(prec5, 1)
    if (i + 1) % args.print_freq == 0:
        print(f' video {i+1} done, total {i+1}/{len(data_loader)}, average {float(cnt_time) / (i+1):.3f} sec/video, moving Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [np.argmax(p) for p in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.val_list)]
    order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    reorder_pred = [None] * len(output)
    output_csv = []
    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        if idx > args.max_num:
            continue

        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]
        reorder_pred[idx] = video_pred[i]
        output_csv.append('%s;%s'%(name_list[i], categories[video_pred[i]]))

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)

    with open(args.save_scores.replace('npz','csv'),'w') as f:
        f.write('\n'.join(output_csv))