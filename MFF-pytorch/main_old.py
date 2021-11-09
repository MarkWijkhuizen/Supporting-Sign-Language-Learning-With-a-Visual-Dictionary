import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data.sampler import SequentialSampler
from multiprocessing import cpu_count
from datetime import datetime
from dataset_custom import TSNDataSet

from models import TSN
from transforms import *
from opts import parser
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import datasets_video
import sys
import os
import easydict
import PIL
import glob
import gc

# empty GPU cache
torch.cuda.empty_cache()

best_prec1 = 0

# Logs DataFrame
TRAIN_LOGS_DF = pd.DataFrame(columns=['epoch', 'train_loss', 'train_prec@1', 'train_prec@5', 'train_prec@100', 'train_prec@1000'])
VAL_LOGS_DF = pd.DataFrame(columns=['epoch', 'val_loss', 'val_prec@1', 'val_prec@5', 'val_prec@100', 'val_prec@1000'])

def main(debug=False, resume=False):
    # empty cache
    torch.cuda.empty_cache()

    print_info() # print provided arguments
    global args, best_prec1
    
    # use arguments as dictionary with set functionality
    args = easydict.EasyDict(vars(parser.parse_known_args()[0]))

    # check dataset is given
    assert args.dataset is not None

    # load correct custom dataset to work with pickle files
    fps_postfix = f'_{args.fps}fps' if args.fps is not None else ''
    global num_classes
    if args.dataset == 'jester':
        index_label = False
        num_classes = 27
        args.epochs = 45
        args.lr_steps = [25, 40]
        args.weight_decay = 1e-1
    elif args.dataset == 'ngt':
        num_classes = args.dataset_subset if args.dataset_subset is not None else 3846
        index_label = args.dataset_subset is not None
        args.epochs = 100
        args.lr_steps = [40, 80]
        args.weight_decay = 4e-1

    # Check if description is set
    assert args.description is not None

    # manually set training parameters
    args.gpus = list(range(torch.cuda.device_count()))
    args.n_frames = args.num_segments * (args.num_motion * 2 + 3)
    args.mffs_shape = (args.n_frames, args.input_size, args.input_size)
    
    # resume checkpoint
    if args.resume:
        args.resume = f'pretrained_models/MFF_jester_RGBFlow_BNInception_segment{args.num_segments}_3f1c_best.pth.tar'

    model = TSN(
        num_classes, args.num_segments, args.modality,
        base_model=args.arch,
        consensus_type=args.consensus_type,
        dropout=args.dropout, num_motion=args.num_motion,
        img_feature_dim=args.img_feature_dim,
        partial_bn=not args.no_partialbn,
        dataset=args.dataset
    )

    input_size = model.input_size
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    n_frames = args.num_motion * 2 + 3

    print(f'mean: {model.input_mean}, std: {model.input_std}, num_classes: {num_classes}')
    print(model)
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        print(args.resume)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(f'=> loaded checkpoint {args.resume} (epoch {args.start_epoch}) (best_prec1 {best_prec1})')
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            return

    cudnn.benchmark = True

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    elif args.modality == 'RGBFlow':
        data_length = args.num_motion

    # VALIDATION DATASET
    # print('Reading Validation Images')
    # val_loader = torch.utils.data.DataLoader(
    #             TSNDataSet(
    #                 f'{args.dataset}_4-MFFs-3f1c{fps_postfix}_val',
    #                 args.mffs_shape,
    #                 torchvision.transforms.Compose([
    #                 # Normalize according to ImageNet means and std
    #                 GroupNormalize(input_rescale, input_mean, input_std, args.n_frames),
    #                 # To Torch Format
    #                 ToTorch(),
    #                ]),
    #                is_train=False, preload_pickles=False, debug=False, n=args.dataset_subset, index_label=index_label,
    #             ),
    #         batch_size=args.batch_size,
    #         shuffle=False,
    #         num_workers=args.workers,
    #         pin_memory=False,
    #     )
    
    
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    
    if args.optimizer == 'SGD':
        print(f'Using SGD Optimizer')
        optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        print('Using Adam Optimizer')
        optimizer = torch.optim.Adam(policies, args.lr, weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, 0)
        return
    
    # TRAINING DATASET
    print('Reading Training Images')
    train_loader = torch.utils.data.DataLoader(
            TSNDataSet(
                f'{args.dataset}_4-MFFs-3f1c{fps_postfix}_train',
                args.mffs_shape,
                torchvision.transforms.Compose([
                    # # Crop or Pad the image and resize to target size
                    # CropPad(20, 224),
                    # # Rotates the image 20 degrees
                    # GroupMultiScaleRotate(20, (args.n_frames, args.input_size, args.input_size)),
                    # # GridMask image masking
                    # GridMask(224),
                    # # Normalize according to ImageNet means and std
                    # GroupNormalize(input_rescale, input_mean, input_std, args.n_frames),
                    # To Torch Format
                    # ToTorch(),
                    Stack(roll=(args.arch in ['BNInception','InceptionV3']), isRGBFlow = (args.modality == 'RGBFlow')),
                    ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                    GroupNormalizeOriginal(input_mean, input_std),
                ]),
                is_train=False, preload_pickles=False, debug=False, n=args.dataset_subset, index_label=index_label,
            ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )

    # check if train images are found, configuration test
    assert len(train_loader) > 0
    
    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    print(f'TRAINING FOR {args.epochs} EPOCHS with BATCH SIZE {args.batch_size}')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch, debug=debug)
        train_original(train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if False and len(val_loader) > 0 and ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1):
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

    # show and save logs
    print('----- TRAINING HISTORY LOGS -----')
    print(TRAIN_LOGS_DF)
    
    print('----- VALIDATION HISTORY LOGS -----')
    print(VAL_LOGS_DF)

    LOGS_DF = TRAIN_LOGS_DF.merge(VAL_LOGS_DF, how='inner', on='epoch')
    LOGS_DF.to_csv(f'output/{args.dataset}_{args.description}_{datetime.now().strftime("%d-%m-%Y_%I%p")}.csv', index=False)

def train(train_loader, model, criterion, optimizer, epoch, debug=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top100 = AverageMeter()
    top1000 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (imgs, lbls) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        lbls = lbls.cuda()
        imgs_var = Variable(imgs)
        lbls_var = Variable(lbls)

        # batch debug
        if False:
            imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
            print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
            lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
            print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)
            test = np.array(imgs)[0, 6:9]
            test = np.moveaxis(test, 0, 2)
            test = (test + [104, 117, 128]).astype(np.uint8)
            print(test.shape)
            Image.fromarray(test).show()
            sys.exit()

        # compute output
        output = model(imgs_var)
        loss = criterion(output, lbls_var)
        
        if i is 0 and debug:
            print(f'output: {output.cpu().detach().numpy().shape}')
            print(f'loss: {loss}')
            print(f'loss: {loss}')
            
        # measure accuracy and record loss
        prec1, prec5, prec100, prec1000 = accuracy(output.data, lbls, topk=(1,5, 100, 1000))
        if i is 0 and debug:
            print(f'prec1: {prec1}, prec5: {prec5}, prec100: {prec100}, prec1000: {prec1000}')
        
        losses.update(loss, imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))
        top100.update(prec100, imgs.size(0))
        top1000.update(prec1000, imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    sys.stdout.flush()
    if epoch == 0:
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)

    print(
            'Train Epoch: [{0}/{1}][{2}/{3}], lr: {lr:.5f}\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\tPrec@5 {top5.val:.2f} ({top5.avg:.2f})\t'
            'Prec@100 {top100.val:.2f} ({top100.avg:.2f})\tPrec@1000 {top1000.val:.2f} ({top1000.avg:.2f})'
        .format(
            epoch + 1, args.epochs, (i+1), len(train_loader),
            batch_time=batch_time, data_time=data_time,
            loss=losses, top1=top1, top5=top5, top100=top100, top1000=top1000, lr=optimizer.param_groups[-1]['lr']
        )
    )

    # Add to logs
    global TRAIN_LOGS_DF
    TRAIN_LOGS_DF = TRAIN_LOGS_DF.append({
        'epoch': epoch,
        'train_loss': losses.avg.cpu().detach().numpy(),
        'train_prec@1': top1.avg.cpu().detach().numpy(),
        'train_prec@5': top5.avg.cpu().detach().numpy(),
        'train_prec@100': top100.avg.cpu().detach().numpy(),
        'train_prec@1000': top1000.avg.cpu().detach().numpy(),
    }, ignore_index=True)

def train_original(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(tqdm(train_loader)):

        # batch debug
        if False:
            imgs = input
            lbls = target
            imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
            print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
            lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
            print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)
            test = np.array(imgs)[0, 6:9]
            test = np.moveaxis(test, 0, 2)
            test = (test + [104, 117, 128]).astype(np.uint8)
            print(test.shape)
            Image.fromarray(test).show()
            sys.exit()

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
                # print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()            

    sys.stdout.flush()

    output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
    print(output)
    log.write(output + '\n')
    log.flush()

    if epoch == 0:
        imgs = input
        lbls = target
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)

            
def validate(val_loader, model, criterion, iter, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top100 = AverageMeter()
    top1000 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(tqdm(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)

            lbls = lbls.cuda()

            # compute output
            output = model(imgs)
                
            loss = criterion(output, lbls)

            # measure accuracy and record loss
            prec1, prec5, prec100, prec1000 = accuracy(output.data, lbls, topk=(1,5, 100, 1000))

            losses.update(loss, imgs.size(0))
            top1.update(prec1, imgs.size(0))
            top5.update(prec5, imgs.size(0))
            top100.update(prec100, imgs.size(0))
            top1000.update(prec1000, imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    sys.stdout.flush()
    if epoch == 0:
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)
    
    print(
            'Val Epoch: [{0}/{1}][{2}/{3}], \t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\tPrec@5 {top5.val:.2f} ({top5.avg:.2f})\t'
            'Prec@100 {top100.val:.2f} ({top100.avg:.2f})\tPrec@1000 {top1000.val:.2f} ({top1000.avg:.2f})'
        .format(
            epoch + 1, args.epochs, (i+1), len(val_loader),
            batch_time=batch_time, data_time=batch_time,
            loss=losses, top1=top1, top5=top5, top100=top100, top1000=top1000,
        )
    )

        # Add to logs
    global VAL_LOGS_DF
    VAL_LOGS_DF = VAL_LOGS_DF.append({
        'epoch': epoch,
        'val_loss': losses.avg.cpu().detach().numpy(),
        'val_prec@1': top1.avg.cpu().detach().numpy(),
        'val_prec@5': top5.avg.cpu().detach().numpy(),
        'val_prec@100': top100.avg.cpu().detach().numpy(),
        'val_prec@1000': top1000.avg.cpu().detach().numpy(),
    }, ignore_index=True)

    gc.collect()

    return top1.avg

def print_info():
    print(f'Python version: {sys.version}')
    print(f'Torch version: {torch.__version__}')
    print(f'Torchvision version: {torchvision.__version__}')
    print(f'Pytroch Cuda Enabled: {torch.cuda.is_available()}')
    # print GPU used
    if torch.cuda.is_available():
        print(f'Pytorch CUDA version: {torch.version.cuda}')
        print(f'Pytorch Number of GPU\'s: {torch.cuda.device_count()}')
        for gpu_device_id in range(torch.cuda.device_count()):
            print(f'Pytroch Cuda Device {gpu_device_id} Name: {torch.cuda.get_device_name(gpu_device_id)}')
    
    # Options
    arguments = easydict.EasyDict(vars(parser.parse_known_args()[0]))
    print('----- PASSED ARGUMENTS -----')
    for key, value in arguments.items():
        print(f'{key}: {value}')

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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.5 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(num_classes, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    name = f'{args.dataset}_{args.description}_{datetime.now().strftime("%d-%m-%Y")}'
    torch.save(state, f'{args.root_model}/{name}_checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(f'{args.root_model}/{name}_checkpoint.pth.tar', f'{args.root_model}/{name}_best.pth.tar')
                
if __name__ == '__main__':
    main(debug=False, resume=False)