import warnings
warnings.simplefilter("ignore", UserWarning)

import argparse
import os
import time
import shutil
import torch
import torchvision
import sys
import torch.nn.parallel
import torch.optim
import easydict
import gc
import pickle

import torch.backends.cudnn as cudnn
import pandas as pd
import torch_optimizer as optim

from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import ndcg_score

# from dataset import TSNDataSet
from dataset_custom import TSNDataSet
from models import TSN
from transforms import *
from opts import parser
import datasets_video
from tqdm import tqdm
from apex import amp
from functools import partialmethod

torch.cuda.empty_cache()

best_prec1 = 0

# NGT Categories and Count
DF_NGT_CAT = pd.read_pickle('df_ngt_cat.pkl.xz')

# Global Arguments
args = easydict.EasyDict(vars(parser.parse_known_args()[0]))

# Silence TQDM
if args.silence_tqdm:
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Aspects
ASPECTS = ['handedness', 'location', 'movement', 'strong_hand', 'weak_hand']

# Logs DataFrame
def get_cols(postfix):
    cols = ['epoch']
    topks = [1, 5, 10, 20]
    # Precision
    if not args.multi_sim_label:
        for p in topks:
            cols.append(f'{postfix}_prec@{p}')
    # Per Aspect Loss
    if args.multi_sim_label:
        for key in ASPECTS:
            cols.append(f'{postfix}_{key}_loss')
    # NDCG
    if args.ndcg:
        for k in topks:
            cols.append(f'{postfix}_NDCG@{k}')
    if args.loc_cat:
        for k in topks:
            cols.append(f'{postfix}_location_p@{k}')
    if args.mov_cat:
        for k in topks:
            cols.append(f'{postfix}_movement_p@{k}')
    if args.ndcg_detailed or args.multi_sim_label:
        for key in ASPECTS:
            for k in topks:
                cols.append(f'{postfix}_{key}_NDCG@{k}')
    print(f'{postfix} cols: {cols}')
    return cols

TRAIN_LOGS_DF = pd.DataFrame(columns=get_cols('train'))
VAL_LOGS_DF = pd.DataFrame(columns=get_cols('val'))
# VAL_LOGS_DF2 = pd.DataFrame(columns=['epoch', 'val2_loss', 'val2_prec@1', 'val2_prec@5', 'val2_prec@100', 'val2_prec@1000'])
pd.options.display.max_rows = 999

def main():
    global args, best_prec1, num_class, USE_AMP
    

    USE_AMP = args['amp']
    if USE_AMP:
        print('--- USING AUTOMATIX MIXED PRECISION ---')

    # check if all log folders are created
    check_rootfolders()
    # print training arguments
    print_info()

    # train log
    log_training = open(os.path.join(args.root_log, '%s.csv' % args.store_name), 'w')
    
    # Configuration
    fps_postfix = f'_{args.fps}fps' if args.fps else ''
    if 'jester' in args.dataset:
        num_class = 27
    elif 'ngt' in args.dataset:
        num_class = args.num_class
    
    print(f'N_CLASS: {num_class}, INDEX_LABEL: {args.index_label}')

    # Optional Optical Flow Frame Postfix
    if args.num_motion_subset is not None:
        args.num_motion_model = args.num_motion_subset
        print(f'!!! Overriding number of Optical Flow Frames from {args.num_motion} to {args.num_motion_subset}')
    else:
        args.num_motion_model = args.num_motion

    args.store_name = f'MFF_{args.dataset}_{args.modality}_{args.arch}_segment{args.num_segments}_{args.num_motion}f1c_{args.description}'
    print('storing name: ' + args.store_name)

    # Optical Flow Key Frame Extraction Postfix
    if args.of_kfe:
        of_kfe_postfix = '_of_kfe'
    elif args.op_kfe:
        of_kfe_postfix = '_op_kfe'
    else:
        of_kfe_postfix = ''

    # Use Softmax Output Activation
    args.softmax = not args.sim_label and not args.multi_sim_label

    model = TSN(
            args, base_model=args.arch, dropout=args.dropout,
            num_motion=args.num_motion_model, partial_bn=not args.no_partialbn,
            softmax=args.softmax,
        )

    # print model head
    print_model_head(model)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    input_rescale = model.rescale
    model_name = model.name

    # manually set training parameters
    args.gpus = list(range(torch.cuda.device_count()))
    args.mff_frames_dataset = args.num_motion * 2 + 3
    args.mff_frames_model = args.num_motion_model * 2 + 3
    args.n_frames_dataset = args.num_segments * (args.num_motion * 2 + 3)
    args.n_frames_model = args.num_segments * (args.num_motion_model * 2 + 3)
    args.mffs_shape_dataset = (args.n_frames_dataset, args.input_size, args.input_size)
    args.mffs_shape_model = (args.n_frames_model, args.input_size, args.input_size)

    policies = model.get_optim_policies()

    cudnn.benchmark = True

    # Attributes
    train_attr= ['label']
    if args.sim_label or args.ndcg:
        train_attr.append('similarity')
    if args.loc_cat:
        train_attr += ['location cat 1', 'location cat count 1', 'location cat 2', 'location cat count 2']
    if args.mov_cat:
        train_attr += ['movement cat', 'movement cat count']
    if args.ndcg_detailed:
        train_attr += ['handedness_similarity', 'location_similarity', 'movement_similarity', 'strong_hand_similarity', 'weak_hand_similarity']

    print(f'The following Train attributes are used: {train_attr}')


    # TRAINING DATASET
    
    # Train Augmentations
    if args.train_augmentations:
        train_augs = [
            # Crop or Pad the image and resize to target size
            CropPad(20, 224),
            
            # Rotates the image 20 degrees
            GroupMultiScaleRotate(20, (args.n_frames_model, args.input_size, args.input_size), resize_only=False),
        ]
        if args.gridmask:
            # GridMask image masking
            train_augs.append(GridMask(args.mff_frames_model))
    else:
        train_augs = []
    print('Reading Training Images')
    train_loader = torch.utils.data.DataLoader(
            TSNDataSet(
                f'{args.dataset}_{args.num_segments}-MFFs-{args.num_motion}f1c{fps_postfix}{of_kfe_postfix}_fr{args.frame_range}_train',
                args.mffs_shape_dataset, args.mffs_shape_model,
                torchvision.transforms.Compose([
                    # Normalize according to ImageNet means and std
                    GroupNormalize(input_rescale, input_mean, input_std, args.n_frames_model, args.num_segments, args.num_motion_model),
                    *train_augs,
                ]),
                is_train=True, debug=False, n=args.dataset_subset[0], index_label=args.index_label, compression=args.compression,
                dataset_ratio=args.dataset_ratio, num_class=args.num_class, attr=train_attr,
                multi_sim_label=args.multi_sim_label, num_motion_subset=args.num_motion_subset, mff_frames_dataset=args.mff_frames_dataset,
            ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers[0],
        pin_memory=False,
    )

    # VALIDATION DATASET
    val_attr = train_attr.copy()
    if args.person_label:
        val_attr.append('person')

    print(f'The following Validation attributes are used: {val_attr}')

    if args.validate:
        print('Reading Validation 1 Images')
        val_loader = torch.utils.data.DataLoader(
                TSNDataSet(
                    f'{args.dataset}_{args.num_segments}-MFFs-{args.num_motion}f1c{fps_postfix}{of_kfe_postfix}_fr{args.frame_range}_{args.validate}',
                    args.mffs_shape_dataset, args.mffs_shape_model,
                    torchvision.transforms.Compose([
                        # Normalize according to ImageNet means and std
                        GroupNormalize(input_rescale, input_mean, input_std, args.n_frames_model, args.num_segments, args.num_motion_model),
                    ]),
                    is_train=False, debug=False, n=args.dataset_subset[1], index_label=args.index_label,
                    compression=args.compression, cache=args.cache_val, num_class=args.num_class, attr=val_attr,
                    multi_sim_label=args.multi_sim_label, num_motion_subset=args.num_motion_subset, mff_frames_dataset=args.mff_frames_dataset,
                ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers[1],
            pin_memory=False,
        )

        # print('Reading Validation 2 Images')
        # val_loader2 = torch.utils.data.DataLoader(
        #         TSNDataSet(
        #             f'{args.dataset}_4-MFFs-3f1c{fps_postfix}_of_kfe_v2_fr2_{args.validate}',
        #             args.mffs_shape,
        #             torchvision.transforms.Compose([
        #                 # Normalize according to ImageNet means and std
        #                 GroupNormalize(input_rescale, input_mean, input_std, args.n_frames),
        #             ]),
        #             is_train=False, debug=False, n=args.dataset_subset[1], index_label=index_label,
        #         ),
        #     batch_size=args.batch_size,
        #     shuffle=False,
        #     num_workers=args.workers,
        #     pin_memory=False,
        # )

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        if args.sim_label or args.multi_sim_label:
            print(f'Using BCEWithLogitsLoss')
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none').cuda()
        elif args.label_smoothing is not None:
            print(f'Using label smoothing CrossEntropyLoss with smoothing={args.label_smoothing}')
            criterion = LabelSmoothingLoss(args.num_class, smoothing=args.label_smoothing)
        else:
            print(f'Using CrossEntropyLoss')
            criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError('Unknown loss type')

    if args.optimizer == 'SGD':
        print('Using SGD Optimizer')
        optimizer = torch.optim.SGD(policies, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        print('Using AdamW optimizer with amsgrad')
        optimizer = torch.optim.AdamW(policies, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.optimizer == 'AdaBound':
        print('Using AdaBound optimizer')
        optimizer = optim.AdaBound(policies, lr=args.lr, weight_decay=args.weight_decay, amsbound=False)
    else:
        raise ValueError(f'Unknown Optimizer: {args.optimizer}')

    # Model building with Automatic Mixed Precision
    if USE_AMP:
        model.to('cuda')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale=1.0)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    # LOAD MODEL
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'=> Loading Checkpoint {args.resume}')
            checkpoint = torch.load(args.resume)
            args.start_epoch = 0 #checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']

            # AVG consensus type
            if 'consensus.classifier.1.weight' in model.state_dict().keys():
                same_head = checkpoint['state_dict']['new_fc.weight'].shape == model.state_dict()['module.new_fc.weight'].shape
                print(f'AVG Same Head: {same_head}')
                if not same_head:
                    checkpoint['state_dict']['new_fc.weight'] = model.state_dict()['module.new_fc.weight']
                    checkpoint['state_dict']['new_fc.bias'] = model.state_dict()['module.new_fc.bias']
            # MLP consensus type
            else:
                same_head = True
                # Add all layers which are present in the model but not in the checkpoint
                add_keys = []
                for k, v in model.state_dict().items():
                    k_other = k.replace('module.', '')
                    if 'consensus' in k or 'arcmargin' in k:
                        if not (k_other in checkpoint['state_dict'] and v.shape == checkpoint['state_dict'][k_other].shape):
                            same_head = False
                            checkpoint['state_dict'][k_other] = v
                            add_keys.append(k_other)
                
                print(f'Added the following keys to the checkpoint: {add_keys}')
                
                # Remove all layers which are present in checkpoint but not in model
                del_keys = []
                for k, v in checkpoint['state_dict'].items():
                    k_other = f'module.{k}'
                    if k_other  not in model.state_dict():
                        del_keys.append(k)

                print(f'Deleting the following keys from the checkpoint: {del_keys}')
                for k in del_keys:
                    del checkpoint['state_dict'][k]

                print(f'MLP Same Head: {same_head}')

            model.module.load_state_dict(checkpoint['state_dict'])
            print(f'=> Successfully Loaded Checkpoint {args.resume} (epoch {args.start_epoch}) (best_prec1 {best_prec1:.2f})')
        else:
            print(f'=> No Checkpoint Found At {args.resume}')

    # Evaluate Model
    if args.evaluate:
        print(f'Evaluating...')
        validate(args, val_loader, model, criterion, optimizer, 0, log_training)
        return

    # Create MFF's Embeddings
    if args.create_embeddings:
        create_embeddings(val_loader, model, 'val')
        train_loader_test = train_loader = torch.utils.data.DataLoader(
                TSNDataSet(
                    f'{args.dataset}_{args.num_segments}-MFFs-{args.num_motion}f1c{fps_postfix}{of_kfe_postfix}_fr{args.frame_range}_train',
                    args.mffs_shape,
                    torchvision.transforms.Compose([GroupNormalize(input_rescale, input_mean, input_std, args.n_frames_model, args.num_segments, args.num_motion)]),
                    is_train=False, debug=False, n=args.dataset_subset[0], index_label=args.index_label, compression=args.compression,
                    dataset_ratio=args.dataset_ratio, cache=args.cache[0], num_class=args.num_class, attr=train_attr,
                    multi_sim_label=args.multi_sim_label,
                ),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        create_embeddings(train_loader_test, model, 'train')
        return

    # Set -1 lr steps to infinity
    args.lr_steps = [i if i > 0 else np.inf for i in args.lr_steps]

    print(f'Training for {args.epochs} Epochs, LR Steps: {args.lr_steps}')
    # Set best precision to -1
    best_prec1 = -1
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, log_training)

        # evaluate on validation set
        if args.validate and ((epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1):
            prec1 = validate(args, val_loader, model, criterion, optimizer, epoch, log_training)
            # validate(val_loader2, model, criterion, optimizer, epoch, log_training, log_idx=2)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                print(f'new best prec1: {prec1:.2f}, old best_prec1: {best_prec1:.2f}')
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_prec1': prec1,
                'description': args.description,
            }, is_best, args.save_best_weights_only)

    # always save last checkpoint when no validation is done
    if not args.validate:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_prec1': -1,
            'description': args.description,
        }, False)

    # show and save logs
    print('----- TRAINING HISTORY LOGS -----')
    print(TRAIN_LOGS_DF)
    
    if args.validate:
        print('----- VALIDATION HISTORY LOGS -----')
        print(VAL_LOGS_DF)

    if args.validate:
        LOGS_DF = TRAIN_LOGS_DF.merge(VAL_LOGS_DF, how='inner', on='epoch')
        # LOGS_DF = LOGS_DF.merge(VAL_LOGS_DF2, how='inner', on='epoch')
        LOGS_DF.to_excel(f'output/{args.dataset}_{args.description}_{datetime.now().strftime("%d-%m-%Y_%I%p")}.xlsx', index=False)
    else:
        TRAIN_LOGS_DF.to_excel(f'output/{args.dataset}_{args.description}_{datetime.now().strftime("%d-%m-%Y_%I%p")}.xlsx', index=False)

def train(args, train_loader, model, criterion, optimizer, epoch, log):
    TOPKS = [1, 5, 10, 20]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    top20 = AverageMeter()
    # Normalized Discounted Cummulative Gain
    if args.ndcg:
        ndcg_top1 = AverageMeter()
        ndcg_top5 = AverageMeter()
        ndcg_top10 = AverageMeter()
        ndcg_top20 = AverageMeter()
    # Location Category Accuracy
    if args.loc_cat:
        loc_cat_top1 = AverageMeter()
        loc_cat_top5 = AverageMeter()
        loc_cat_top10 = AverageMeter()
        loc_cat_top20 = AverageMeter()
    # Movement Category Accuracy
    if args.mov_cat:
        mov_cat_top1 = AverageMeter()
        mov_cat_top5 = AverageMeter()
        mov_cat_top10 = AverageMeter()
        mov_cat_top20 = AverageMeter()
    # Normalized Discounted Cummulative Gain per Part
    if args.ndcg_detailed or args.multi_sim_label:
        ndcg_detailed_dict = dict()
        ndcg_detailed_keys = ['handedness', 'location', 'movement', 'strong_hand', 'weak_hand']
        for key in ndcg_detailed_keys:
            ndcg_detailed_dict[key] = dict()
            for topk in TOPKS:
                ndcg_detailed_dict[key][f'ndcg@{topk}'] = AverageMeter()
    # Per Aspect Loss
    if args.multi_sim_label:
        aspect_loss = dict()
        for key in ASPECTS:
            aspect_loss[key] = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (imgs, lbls, attr) in enumerate(tqdm(train_loader)):
        batch_size =  imgs.size(0)
        if False: # DEBUG CODE
            img = imgs.numpy()[0,i*9+6:i*9+6+3]
            print('original', img.shape)
            img = np.moveaxis(img, 0, 2)
            if img.shape[2] == 1:
                img = img[:,:,0]
                img = (img * 0.229) + 0.485
            else:
                img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).show()
            sys.exit()

        # measure data loading time
        data_time.update(time.time() - end)

        lbls = lbls.cuda()
        input_var = Variable(imgs)
        if args.sim_label:
            sims = attr['similarity'].cuda()
            target_var = Variable(sims)
        elif args.multi_sim_label:
            target_var_cpu = lbls.permute(1,0,2).cpu().detach().numpy()
            target_var = Variable(lbls.permute(1,0,2))
        else:
            target_var = Variable(lbls)

        # compute output
        output = model(input_var, lbls)
        loss = []
        loss_sum = 0
        if type(output) is list:
            output_sum = np.zeros(shape=[batch_size, args.num_class], dtype=np.float32)
            for o, t in zip(output, target_var):
                l = criterion(o, t)
                loss.append(l.cpu().detach().numpy())
                loss_sum += l
                output_sum += o.cpu().detach().numpy()
        else:
            loss_sum = criterion(output, target_var)
            
            if args.sim_label: # Similarity weighted loss
                loss_weights = 1 - target_var ** args.sim_label_exp
                loss_weights[loss_weights < args.sim_label_eps] = args.sim_label_eps
                loss_weights = 1 / loss_weights
                for lbl_idx, lbl in enumerate(lbls):
                    loss_weights[lbl_idx, lbl] *= args.sim_label_target_scale
                loss_sum = torch.mean(loss_sum * loss_weights)
            else:
                loss_sum = torch.mean(loss_sum)
            
            loss =[loss_sum.cpu().detach().numpy()]

        if not args.multi_sim_label:
            output_np = output.cpu().detach().numpy()

        losses.update(sum(loss), batch_size)

        # Precision At K
        if args.multi_sim_label:
            prec1, prec5, prec10, prec20 = accuracy(torch.tensor(output_sum), attr['label'], topk=TOPKS)
        else:
            prec1, prec5, prec10, prec20 = accuracy(output.data, lbls, topk=TOPKS)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)
        top10.update(prec10, batch_size)
        top20.update(prec20, batch_size)

        # Multi Similarity NDCG
        if args.multi_sim_label:
            for o, t, al, k in zip(output, target_var_cpu, loss, ASPECTS):
                o = o.cpu().detach().numpy()
                ndcg_scores = ndcg(o, t, topk=TOPKS)
                # NDCG
                for topk, ndcg_score in zip(TOPKS, ndcg_scores):
                    ndcg_detailed_dict[k][f'ndcg@{topk}'].update(ndcg_score, batch_size)
                # Aspect Loss
                aspect_loss[k].update(al, batch_size)

        # Discounted Cummulative Gain
        if args.ndcg:
            ndcg1, ndcg5, ndcg10, ndcg20 = ndcg(output.cpu().detach().numpy(), attr['similarity'].cpu().detach().numpy(), topk=TOPKS)
            ndcg_top1.update(ndcg1, batch_size)
            ndcg_top5.update(ndcg5, batch_size)
            ndcg_top10.update(ndcg10, batch_size)
            ndcg_top20.update(ndcg20, batch_size)
        # Location Category Accuracy
        if args.loc_cat:
            lct1, lct5, lct10, lct20 = cat_precision(attr, output_np, 'location', topk=TOPKS)
            loc_cat_top1.update(lct1, batch_size)
            loc_cat_top5.update(lct5, batch_size)
            loc_cat_top10.update(lct10, batch_size)
            loc_cat_top20.update(lct20, batch_size)

        # Movement Category Accuracy
        if args.mov_cat:
            mcp1, mcp5, mcp10, mcp20 = cat_precision(attr, output_np, 'movement', topk=TOPKS)
            mov_cat_top1.update(mcp1, batch_size)
            mov_cat_top5.update(mcp5, batch_size)
            mov_cat_top10.update(mcp10, batch_size)
            mov_cat_top20.update(mcp20, batch_size)

        if args.ndcg_detailed:
            for key in ndcg_detailed_keys:
                ndcg_scores = ndcg(output_np, attr[f'{key}_similarity'].cpu().detach().numpy(), topk=TOPKS)
                for topk, ndcg_score in zip(TOPKS, ndcg_scores):
                    ndcg_detailed_dict[key][f'ndcg@{topk}'].update(ndcg_score, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if USE_AMP:
            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_sum.backward()
        
        # Optimizer step
        optimizer.step()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()        

    sys.stdout.flush()
    training_stats = ('Train Epoch {} lr: {:.5f}\t'
                'Time {:.3f}\t Data {:.3f}\t Loss {:.4f}\t'
                'Prec@1 {:.2f}\t Prec@5 {:.2f} \t Prec@10 {:.2f} \t Prec@20 {:.2f}'
            .format(
                    f'{epoch+1}, {i+1}/{len(train_loader)}'.ljust(12),
                    optimizer.param_groups[-1]['lr'],
                    batch_time.avg, data_time.avg, losses.avg, top1.avg, top5.avg, top10.avg, top20.avg
                )
            )

    # NDCG
    if args.ndcg:
        training_stats += f'\t NDCG@1 {ndcg_top1.avg:.2f}\t NDCG@5 {ndcg_top5.avg:.2f}\t NDCG@10 {ndcg_top10.avg:.2f}\t NDCG@20 {ndcg_top20.avg:.2f}'
    # Location Accuracy Stats
    if args.loc_cat:
        training_stats += f'\t loc_acc@1 {loc_cat_top1.avg:.2f}\t loc_acc@5 {loc_cat_top5.avg:.2f}\t loc_acc@10 {loc_cat_top10.avg:.2f}\t loc_acc@20 {loc_cat_top20.avg:.2f}'
    # Movement Accuracy Stats
    if args.mov_cat:
        training_stats += f'\t mov_acc@1 {mov_cat_top1.avg:.2f}\t mov_acc@5 {mov_cat_top5.avg:.2f}\t mov_acc@10 {mov_cat_top10.avg:.2f}\t mov_acc@20 {mov_cat_top20.avg:.2f}'
    # Aspect Loss
    if args.multi_sim_label:
        for k, v in aspect_loss.items():
            training_stats += f'\t {k}_loss {v.avg:.2f}'
    
    # Normalized Dictouned Cummulative Gain Detailed
    if args.ndcg_detailed or args.multi_sim_label:
        for key in ndcg_detailed_keys:
            ndcg_dict = ndcg_detailed_dict[key]
            for topk in TOPKS:
                training_stats += f'\tndcg_{key}@{topk} {ndcg_dict[f"ndcg@{topk}"].avg:.2f}'

    print(training_stats)
    log.write(training_stats + '\n')
    log.flush()

    # Add to logs
    global TRAIN_LOGS_DF
    TRAIN_LOGS_DF_APPEND = {
        'epoch': epoch + 1,
        'train_loss': losses.avg,
        'train_prec@1': top1.avg.cpu().detach().numpy(),
        'train_prec@5': top5.avg.cpu().detach().numpy(),
        'train_prec@10': top10.avg.cpu().detach().numpy(),
        'train_prec@20': top20.avg.cpu().detach().numpy(),
    }
    if args.ndcg:
        TRAIN_LOGS_DF_APPEND.update({
            'train_NDCG@1': ndcg_top1.avg,
            'train_NDCG@5': ndcg_top5.avg,
            'train_NDCG@10': ndcg_top10.avg,
            'train_NDCG@20': ndcg_top20.avg,
        })
    if args.loc_cat:
        TRAIN_LOGS_DF_APPEND.update({
            'train_location_p@1': loc_cat_top1.avg,
            'train_location_p@5': loc_cat_top5.avg,
            'train_location_p@10': loc_cat_top10.avg,
            'train_location_p@20': loc_cat_top20.avg,
        })
    if args.mov_cat:
        TRAIN_LOGS_DF_APPEND.update({
            'train_movement_p@1': mov_cat_top1.avg,
            'train_movement_p@5': mov_cat_top5.avg,
            'train_movement_p@10': mov_cat_top10.avg,
            'train_movement_p@20': mov_cat_top20.avg,
        })
    # Per Aspect Loss
    if args.multi_sim_label:
        for k, v in aspect_loss.items():
            TRAIN_LOGS_DF_APPEND.update({ f'train_{k}_loss': v.avg })
    # Per Aspect NDCG At K
    if args.ndcg_detailed or args.multi_sim_label:
        for key in ndcg_detailed_keys:
            ndcg_dict = ndcg_detailed_dict[key]
            for topk in TOPKS:
                TRAIN_LOGS_DF_APPEND.update({ f'train_{key}_NDCG@{topk}': ndcg_dict[f'ndcg@{topk}'].avg })
        
    TRAIN_LOGS_DF = TRAIN_LOGS_DF.append(TRAIN_LOGS_DF_APPEND, ignore_index=True)

    if epoch == 0:
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)
        if type(attr) is dict:
            print(f'attr keys: {attr.keys()}')
        if args.multi_sim_label:        
            print(f'output len: {len(output)}')
        else:
            output_stats = output.shape, output.float().mean(), output.float().std(), output.min(), output.max(), output.dtype, type(output)
            print('output shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % output_stats)

def validate(args, val_loader, model, criterion, optimizer, epoch, log, log_idx=1):
    TOPKS = [1, 5, 10, 20]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    top20 = AverageMeter()
    # Normalized Discounted Cummulative Gain
    if args.ndcg:
        ndcg_top1 = AverageMeter()
        ndcg_top5 = AverageMeter()
        ndcg_top10 = AverageMeter()
        ndcg_top20 = AverageMeter()
    # Location Category Accuracy
    if args.loc_cat:
        loc_cat_top1 = AverageMeter()
        loc_cat_top5 = AverageMeter()
        loc_cat_top10 = AverageMeter()
        loc_cat_top20 = AverageMeter()
    # Movement Category Accuracy
    if args.mov_cat:
        mov_cat_top1 = AverageMeter()
        mov_cat_top5 = AverageMeter()
        mov_cat_top10 = AverageMeter()
        mov_cat_top20 = AverageMeter()
    # Normalized Discounted Cummulative Gain per Part
    if args.ndcg_detailed or args.multi_sim_label:
        ndcg_detailed_dict = dict()
        ndcg_detailed_keys = ['handedness', 'location', 'movement', 'strong_hand', 'weak_hand']
        for key in ndcg_detailed_keys:
            ndcg_detailed_dict[key] = dict()
            for topk in TOPKS:
                ndcg_detailed_dict[key][f'ndcg@{topk}'] = AverageMeter()
    # Per Aspect Loss
    if args.multi_sim_label:
        aspect_loss = dict()
        for key in ASPECTS:
            aspect_loss[key] = AverageMeter()

    person_topk = dict()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (imgs, lbls, attr) in enumerate(tqdm(val_loader)):
        batch_size = imgs.size(0)
        if False: # DEBUG CODE
            img = imgs.numpy()[0,i*9+0:i*9+0+1]
            print('original', img.shape)
            img = np.moveaxis(img, 0, 2)
            if img.shape[2] == 1:
                img = img[:,:,0]
                img = (img * 0.229) + 0.485
            else:
                img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).show()
            sys.exit()

        lbls = lbls.cuda()
        input_var = Variable(imgs)
        if args.sim_label:
            sims = attr['similarity'].cuda()
            target_var = Variable(sims)
        elif args.multi_sim_label:
            target_var_cpu = lbls.permute(1,0,2).cpu().detach().numpy()
            target_var = Variable(lbls.permute(1,0,2))
        else:
            target_var = Variable(lbls)
        
        with torch.no_grad():
            # compute output
            output = model(input_var, lbls)

        # compute output
        loss = []
        loss_sum = 0
        if type(output) is list:
            output_sum = np.zeros(shape=[batch_size, args.num_class], dtype=np.float32)
            for o, t in zip(output, target_var):
                l = criterion(o, t)
                loss.append(l.cpu().detach().numpy())
                loss_sum += l
                output_sum += o.cpu().detach().numpy()
        else:
            loss_sum = criterion(output, target_var)
            
            if args.sim_label: # Similarity weighted loss
                loss_weights = 1- target_var ** args.sim_label_exp
                loss_weights[loss_weights < args.sim_label_eps] = args.sim_label_eps
                loss_weights = 1 / loss_weights
                for lbl_idx, lbl in enumerate(lbls):
                    loss_weights[lbl_idx, lbl] *= args.sim_label_target_scale
                loss_sum = torch.mean(loss_sum * loss_weights)
            else:
                loss_sum = torch.mean(loss_sum)
            
            loss =[loss_sum.cpu().detach().numpy()]

        if not args.multi_sim_label:
            output_np = output.cpu().detach().numpy()

        losses.update(sum(loss), batch_size)
        
        # Save results for confusion matrix
        if args.confusion_matrix and args.evaluate:
            concat = np.concat([lbls, output], axis=0)
            print(f'concat shape: {concat.shape}')
            if 'confusion_matrix' not in locals():
                confusion_matrix = concat
                print(f'after confusion_matrix shape: {confusion_matrix.shape}')
            else:
                print(f'before confusion_matrix shape: {confusion_matrix.shape}')
                confusion_matrix = np.concatenate([confusion_matrix, concat], axis=0)
                print(f'after confusion_matrix shape: {confusion_matrix.shape}')

        # measure accuracy and record loss
        if args.person_label:
            for p in np.unique(attr['person']):
                p_idxs = np.where(np.array(attr['person']) == p)[0]
                prec1, prec5, prec10, prec20 = accuracy(output[p_idxs], lbls[p_idxs], topk=TOPKS)
                # save results
                if p not in person_topk:
                    person_topk[p] = dict()
                    for key, prec in [('p@1', prec1), ('p@5', prec5), ('p@10', prec10), ('p@20', prec20)]:
                        person_topk[p][key] = AverageMeter()
                        person_topk[p][key].update(prec, len(p_idxs))
                else:
                    for key, prec in [('p@1', prec1), ('p@5', prec5), ('p@10', prec10), ('p@20', prec20)]:
                        person_topk[p][key].update(prec, len(p_idxs))

        # Precision At K
        if args.multi_sim_label:
            prec1, prec5, prec10, prec20 = accuracy(torch.tensor(output_sum), attr['label'], topk=TOPKS)
        else:
            prec1, prec5, prec10, prec20 = accuracy(output.data, lbls, topk=TOPKS)
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)
        top10.update(prec10, batch_size)
        top20.update(prec20, batch_size)

        # Multi Similarity Label NDCG
        if args.multi_sim_label:
            for o, l, al, k in zip(output, target_var_cpu, loss, ASPECTS):
                o = o.cpu().detach().numpy()
                ndcg_scores = ndcg(o, l, topk=TOPKS)
                for topk, ndcg_score in zip(TOPKS, ndcg_scores):
                    ndcg_detailed_dict[k][f'ndcg@{topk}'].update(ndcg_score, batch_size)
                # Aspect Loss
                aspect_loss[k].update(al, batch_size)

        # Discounted Cummulative Gain
        if args.ndcg:
            ndcg1, ndcg5, ndcg10, ndcg20 = ndcg(output_np, attr['similarity'].cpu().detach().numpy(), topk=TOPKS)
            ndcg_top1.update(ndcg1, batch_size)
            ndcg_top5.update(ndcg5, batch_size)
            ndcg_top10.update(ndcg10, batch_size)
            ndcg_top20.update(ndcg20, batch_size)

        # Save results for confusion matrix
        if args.save_ndcg and args.evaluate:
            # Unnormalised NDCG scores
            ndcg1_nn, ndcg5_nn, ndcg10_nn, ndcg20_nn = ndcg(output_np, attr['similarity'].cpu().detach().numpy(), topk=TOPKS, normalise=False)
            if 'ndcg_matrix' not in locals():
                ndcg_matrix = { 'ndcg@1': ndcg1_nn, 'ndcg@5': ndcg5_nn, 'ndcg@10': ndcg10_nn, 'ndcg@20': ndcg20_nn }
            else:
                for (k, v), concat in zip(ndcg_matrix.items(), [ndcg1_nn, ndcg5_nn, ndcg10_nn, ndcg20_nn]):
                    ndcg_matrix[k] += concat

        # Location Category Accuracy
        if args.loc_cat:
            lct1, lct5, lct10, lct20 = cat_precision(attr, output_np, 'location', topk=TOPKS)
            loc_cat_top1.update(lct1, batch_size)
            loc_cat_top5.update(lct5, batch_size)
            loc_cat_top10.update(lct10, batch_size)
            loc_cat_top20.update(lct20, batch_size)

        # Movement Category Accuracy
        if args.mov_cat:
            mcp1, mcp5, mcp10, mcp20 = cat_precision(attr, output_np, 'movement', topk=TOPKS)
            mov_cat_top1.update(mcp1, batch_size)
            mov_cat_top5.update(mcp5, batch_size)
            mov_cat_top10.update(mcp10, batch_size)
            mov_cat_top20.update(mcp20, batch_size)

        if args.ndcg_detailed:
            for key in ndcg_detailed_keys:
                ndcg_scores = ndcg(output_np, attr[f'{key}_similarity'].cpu().detach().numpy(), topk=TOPKS)
                for topk, ndcg_score in zip(TOPKS, ndcg_scores):
                    ndcg_detailed_dict[key][f'ndcg@{topk}'].update(ndcg_score, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Save NDCG results
    if args.save_ndcg:
        model_checkpoint_name = args.resume.split("/")[1].split('.')[0]
        with open(f'ndcg_{model_checkpoint_name}.pkl', 'wb') as f:
                pickle.dump(ndcg_matrix, f)

    # Global Validation Stats
    sys.stdout.flush()
    val_stats = ('Val{}  Epoch {} lr: {:.5f}\t'
                'Time {:.3f}\t Data {:.3f}\t Loss {:.4f}\t'
                'Prec@1 {:.2f}\t Prec@5 {:.2f} \t Prec@10 {:.2f} \t Prec@20 {:.2f}'
            .format(
                    log_idx, f'{epoch+1}, {i+1}/{len(val_loader)}'.ljust(12), optimizer.param_groups[-1]['lr'],
                    batch_time.avg, data_time.avg, losses.avg,
                    top1.avg, top5.avg, top10.avg, top20.avg,
                )
            )
    # NDCG Stats
    if args.ndcg:
        val_stats += f'\tNDCG@1 {ndcg_top1.avg:.2f}\t NDCG@5 {ndcg_top5.avg:.2f}\t NDCG@10 {ndcg_top10.avg:.2f}\t NDCG@20 {ndcg_top20.avg:.2f}'
    # Location Accuracy Stats
    if args.loc_cat:
        val_stats += f'\tloc_acc@1 {loc_cat_top1.avg:.2f}\t loc_acc@5 {loc_cat_top5.avg:.2f}\t loc_acc@10 {loc_cat_top10.avg:.2f}\t loc_acc@20 {loc_cat_top20.avg:.2f}'
    # Movement Accuracy Stats
    if args.mov_cat:
        val_stats += f'\tmov_acc@1 {mov_cat_top1.avg:.2f}\t mov_acc@5 {mov_cat_top5.avg:.2f}\t mov_acc@10 {mov_cat_top10.avg:.2f}\t mov_acc@20 {mov_cat_top20.avg:.2f}'
    # Aspect Loss
    if args.multi_sim_label:
        for k, v in aspect_loss.items():
            val_stats += f'\t {k}_loss {v.avg:.2f}'
    # Normalized Dictouned Cummulative Gain Detailed
    if args.ndcg_detailed or args.multi_sim_label:
        for key in ndcg_detailed_keys:
            ndcg_dict = ndcg_detailed_dict[key]
            for topk in TOPKS:
                val_stats += f'\tndcg_{key}@{topk} {ndcg_dict[f"ndcg@{topk}"].avg:.3f}'

    print(val_stats)
    log.write(val_stats + '\n')
    log.flush()

    # Per person validation stats
    if args.person_label:
        for p, topks in person_topk.items():
            print(f'{p.ljust(15, " ")} | ', end='')
            for idx, (key, prec) in enumerate(topks.items()):
                if idx == 0:
                    print(f'count: {prec.count}, ', end='')
                elif key is not 'prec@20':
                    print(', ', end='')
                print(f'\t{key}: {prec.avg:.2f}', end='')
            print()

    # Add to logs
    if log_idx == 1:
        global VAL_LOGS_DF
        VAL_LOGS_DF_APPEND = {
            'epoch': epoch + 1,
            'val_loss': losses.avg,
            'val_prec@1': top1.avg.cpu().detach().numpy(),
            'val_prec@5': top5.avg.cpu().detach().numpy(),
            'val_prec@10': top10.avg.cpu().detach().numpy(),
            'val_prec@20': top20.avg.cpu().detach().numpy(),
        }
        if args.ndcg:
            VAL_LOGS_DF_APPEND.update({
                'val_NDCG@1': ndcg_top1.avg,
                'val_NDCG@5': ndcg_top5.avg,
                'val_NDCG@10': ndcg_top10.avg,
                'val_NDCG@20': ndcg_top20.avg,
            })
        if args.loc_cat:
            VAL_LOGS_DF_APPEND.update({
                'val_location_p@1': loc_cat_top1.avg,
                'val_location_p@5': loc_cat_top5.avg,
                'val_location_p@10': loc_cat_top10.avg,
                'val_location_p@20': loc_cat_top20.avg,
            })
        if args.mov_cat:
            VAL_LOGS_DF_APPEND.update({
                'val_movement_p@1': mov_cat_top1.avg,
                'val_movement_p@5': mov_cat_top5.avg,
                'val_movement_p@10': mov_cat_top10.avg,
                'val_movement_p@20': mov_cat_top20.avg,
            })
        # Per Aspect Loss
        if args.multi_sim_label:
            for k, v in aspect_loss.items():
                VAL_LOGS_DF_APPEND.update({ f'val_{k}_loss': v.avg })
        # Per Aspect NDCG At K
        if args.ndcg_detailed or args.multi_sim_label:
            for key in ndcg_detailed_keys:
                ndcg_dict = ndcg_detailed_dict[key]
                for topk in TOPKS:
                    VAL_LOGS_DF_APPEND.update({ f'val_{key}_NDCG@{topk}': ndcg_dict[f'ndcg@{topk}'].avg })

        VAL_LOGS_DF = VAL_LOGS_DF.append(VAL_LOGS_DF_APPEND, ignore_index=True)
    else:
        global VAL_LOGS_DF2
        VAL_LOGS_DF2 = VAL_LOGS_DF2.append({
            'epoch': epoch + 1,
            'val2_loss': losses.avg.cpu().detach().numpy(),
            'val2_prec@1': top1.avg.cpu().detach().numpy(),
            'val2_prec@5': top5.avg.cpu().detach().numpy(),
            'val2_prec@10': top10.avg.cpu().detach().numpy(),
            'val2_prec@20': top20.avg.cpu().detach().numpy(),
        }, ignore_index=True)

    if epoch == 0:
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)
        if type(attr) is dict:
            print(f'attr keys: {attr.keys()}')
        if args.multi_sim_label:
            print(f'output len: {len(output)}')
        else:
            output_stats = output.shape, output.float().mean(), output.float().std(), output.min(), output.max(), output.dtype, type(output)
            print('output shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % output_stats)

    # Return best model criterion
    if args.best_k == 1:
        return top1.avg
    elif args.best_k == 20:
        return top20.avg


def save_checkpoint(state, is_best, save_best_weights_only, filename='checkpoint.pth.tar'):
    # Check if only best weights should be saved
    if not save_best_weights_only:
        torch.save(state, f'{args.root_model}/{args.store_name}_checkpoint.pth.tar')
    if is_best:
        torch.save(state, f'{args.root_model}/{args.store_name}_best.pth.tar')
        # shutil.copyfile(f'{args.root_model}/{args.store_name}_checkpoint.pth.tar',f'{args.root_model}/{args.store_name}_best.pth.tar')

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


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

    return lr

def ndcg(output, similarities, topk=(1,), normalise=True):
    res = []
    # Order softmax output
    idxs_best = np.argsort(similarities, axis=1)
    idxs_best = np.flip(idxs_best, axis=1)
    idxs_pred = np.argsort(output, axis=1)
    idxs_pred = np.flip(idxs_pred, axis=1)
    for k in topk:
        topk_res = []
        for out, sims, idxs_b, idxs_p in zip(output, similarities, idxs_best, idxs_pred):
            best = sims[idxs_b]
            pred = sims[idxs_p]
            topk_res.append(ndcg_score([best], [pred], k=k))

        if normalise:
            res.append(sum(topk_res) / len(output))
        else:
            res.append(topk_res)
    return res

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(num_class, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def cat_precision(attr, output, kind, topk):
    cat_topks = []
    pred_idxs_sort = np.argsort(output, axis=1)
    pred_idxs_sort = np.flip(pred_idxs_sort, axis=1)
    if kind == 'location':
        keys = [('location cat 1', 'location cat count 1'), ('location cat 2', 'location cat count 2')]
    elif kind == 'movement':
        keys = [('movement cat', 'movement cat count')]

    for k in topk:
        res = 0
        div = 0
        for idx, idx_sort in enumerate(pred_idxs_sort):
            for ck, cck in keys:
                c = attr[ck][idx]
                cc = attr[cck][idx]
                cps = DF_NGT_CAT.loc[idx_sort, ck].values
                ccps = DF_NGT_CAT.loc[idx_sort, cck].values
                # print(f'c: {c}, cc: {cc}, idx_sort: {idx_sort}, cps: {cps}, ccps: {ccps}')
                for cp in cps[:k]:
                    # update div
                    div += 1 if c != '-' else 0
                    if c == cp and c != '-':
                        add = 1 / (float(min(cc, k)) / float(k) * len(keys)) * 100
                        res += add
                        if k == 1:
                            pass
                            # print('equal', c, cp, f'div: {div}, add: {add}')
                        # print(f'ERROR: cc: {cc}, c: {c}, cp: {cp}')
        # append topk result
        cat_topks.append(res / div)
        
    return cat_topks

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model, args.root_output]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

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
    # print arguments
    print('======== COMMNAND LINE ARGUMENTS ========')
    for k, v in args.items():
        print(f'-- {k.ljust(15)} \t{v}')
    print('=========================================')

def flatten_layer(l):
    res = []
    children = list(l.children())
    if len(children) == 0:
        return [l]
    else:
        for c in children:
            res += flatten_layer(c)
        return res

def print_model_head(model):
    n_print_head_layers = 15
    print('=' * 50)
    print(f'Modified Model Head last {n_print_head_layers} layers')
    # get last layer, can be nested
    children = flatten_layer(model)
   
    # print head in reverse
    for layer_idx, layer in enumerate(children[-n_print_head_layers:]):
        print(f'{n_print_head_layers - layer_idx} --- ', layer)

    print('=' * 50)

def create_embeddings(dataset, model, prefix):
    for i, (imgs, lbls, attr) in enumerate(tqdm(dataset)):
        
        with torch.no_grad():
            embeddings_batch = model(Variable(imgs)).cpu()
        
        if i == 0:
            embeddings = embeddings_batch
            embeddings_lbls = lbls
        else:
            embeddings = np.concatenate((embeddings, embeddings_batch))
            embeddings_lbls = np.concatenate((embeddings_lbls, lbls))
    
    print(f'embeddings shape: {embeddings.shape}, embeddings_lbls shape: {embeddings_lbls.shape}')

    np.save(f'{prefix}_embeddings.npy', embeddings)
    np.save(f'{prefix}_embeddings_lbls.npy', embeddings_lbls)

# Label Smoothing BCE
# Source: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if __name__ == '__main__':
    main()