import os
import torch
import torchvision
import torch.backends.cudnn as cudnn
import pandas as pd

from models import TSN
from dataset_custom import TSNDataSet
from torch.autograd import Variable
from transforms import *
from PIL import Image
from datetime import datetime

num_class = 27

def main():
    
    # Signs of which 2 are visually similar and 1 is not
    CLOSE_FAR_DICT = tuple([
        {
            'close_a': { 'label': 3254, 'folder_idx': 3301, 'sign': 'TURKIJE-A', }, # TURKIJE-A 2243
            'close_b': {'label': 907, 'folder_idx': 914, 'sign': 'DUITSLAND-A', }, # DUITSLAND-A 2158
            'far': {'label': 2086, 'folder_idx': 2107, 'sign': 'MOE-B', }, # MOE-B 343
        }, {
            'close_b': { 'label': 2293, 'folder_idx': 2332, 'sign': 'ONTMOETEN', }, # ONTMOETEN 1337
            'close_a': { 'label': 2333, 'folder_idx': 2372, 'sign': 'OORLOG-A', },  # OORLOG-A 1353
            'far': { 'label': 2913, 'folder_idx': 2959, 'sign': 'SLIM-C', }, # SLIM-C 1542
        }, {
            'close_a': { 'label': 2118, 'folder_idx': 2140, 'sign': 'MOSLIM', }, # MOSLIM 2675
            'close_b': { 'label': 3527, 'folder_idx': 3576, 'sign': 'VOORKOMEN', }, # VOORKOMEN 938
            'far': { 'label': 3181, 'folder_idx': 3228, 'sign': 'THEE', }, # THEE 330
        }, {
            'close_a': { 'label': 3181, 'folder_idx': 3228, 'sign': 'THEE', }, # THEE 330
            'close_b': { 'label': 2961, 'folder_idx': 3007, 'sign': 'SPAREN-A', }, # SPAREN-A 3396
            'far': { 'label': 2726, 'folder_idx': 2770, 'sign': 'RIJLES' }, # RIJLES 8
        }, {
            'close_a': { 'label': 1817, 'folder_idx': 1837, 'sign': 'LAATSTE-A',  }, # LAATSTE-A 5
            'close_b': { 'label': 1818, 'folder_idx': 1838, 'sign': 'LAATSTE-B' }, # LAATSTE-B 6
            'far': { 'label': 3391, 'folder_idx': 3439, 'sign': 'VERGETEN-A' }, # VERGETEN-A 1961
        }, {
            'close_a': { 'label': 2411, 'folder_idx': 2453, 'sign': 'OUDER-B', }, # OUDER-B 1802
            'close_b': { 'label': 591, 'folder_idx': 593, 'sign': 'BOOS-B', }, # BOOS-B 387
            'far': { 'label': 1857, 'folder_idx': 1877, 'sign': 'LEKKER-A', }, # LEKKER-A 2541
        }, {
            'close_a': { 'label': 3178, 'folder_idx': 3225, 'sign': 'TEVREDEN-A', }, # TEVREDEN-A 609
            'close_b': { 'label': 943, 'folder_idx': 953, 'sign': 'EINDELIJK-A', }, # EINDELIJK-A 549
            'far': { 'label': 2446, 'folder_idx': 2488, 'sign': 'PAARS', }, # PAARS 998
        }, {
            'close_a': { 'label': 1265, 'folder_idx': 1281, 'sign': 'GROEN', }, # GROEN 2048
            'close_b': { 'label': 2183, 'folder_idx': 2211, 'sign': 'NIEUW-C' }, # NIEUW-C 2052
            'far': { 'label': 2756, 'folder_idx': 2800, 'sign': 'ROOD', }, # ROOD 378
        }, {
            'close_a': { 'label': 630, 'folder_idx': 635, 'sign': 'BROOD-A', }, # BROOD-A 1159
            'close_b': { 'label': 508, 'folder_idx': 509, 'sign': 'BEWIJS', }, # BEWIJS 412
            'far': { 'label': 489, 'folder_idx': 490, 'sign': 'BESTELLEN-C', }, # BESTELLEN-C 95
        }, {
            'close_a': { 'label': 576, 'folder_idx': 577, 'sign': 'BOEK', }, # BOEK 2850
            'close_b': { 'label': 293, 'folder_idx': 293, 'sign': 'ALSTUBLIEFT', } , # ALSTUBLIEFT 2511
            'far': { 'label': 3260, 'folder_idx': 3307, 'sign': 'TWEEEN-C', } , # TWEEEN-C 2088
        },
        # 10 - 25
        {
            'close_a': { 'label': 32, 'folder_idx': 32, 'sign': '1-A', }, # 1-A 2442
            'close_b': { 'label': 34, 'folder_idx': 34, 'sign': '1.ORD', } , # 1.ORD 2415
            'far': { 'label': 770, 'folder_idx': 775, 'sign': 'DENEMARKEN', } , # DENEMARKEN 1674
        }, {
            'close_a': { 'label': 146, 'folder_idx': 146, 'sign': '8-A', }, # 8-A 2453
            'close_b': { 'label': 137, 'folder_idx': 137, 'sign': '7-A', } , # 7-A 2451
            'far': { 'label': 1296, 'folder_idx': 1312, 'sign': 'GROOTOUDER-A', } , # GROOTOUDER-A 574 
        }, {
            'close_a': { 'label': 1011, 'folder_idx': 1024, 'sign': 'EUROPA-A', }, # EUROPA-A 2233
            'close_b': { 'label': 1012, 'folder_idx': 1025, 'sign': 'EUROPA-B', } , # EUROPA-B 2234
            'far': { 'label': 2346, 'folder_idx': 2385, 'sign': 'OPA', } , # OPA 2186
        }, {
            'close_a': { 'label': 2228, 'folder_idx': 2264, 'sign': 'OCHTEND-A', }, # OCHTEND-A 1437
            'close_b': { 'label': 363, 'folder_idx': 363, 'sign': 'AVOND-A', } , # AVOND-A 1352
            'far': { 'label': 1752, 'folder_idx': 1772, 'sign': 'KOFFIE-B', } , # KOFFIE-B 2540
        }, {
            'close_a': { 'label': 363, 'folder_idx': 363, 'sign': 'AVOND-A', }, # AVOND-A 1352
            'close_b': { 'label': 2319, 'folder_idx': 2358, 'sign': 'ONZICHTBAAR-D', } , # ONZICHTBAAR-D 3018
            'far': { 'label': 1755, 'folder_idx': 1775, 'sign': 'KOKEN-B', } , # KOKEN-B 1174
        }, {
            'close_a': { 'label': 1659, 'folder_idx': 1678, 'sign': 'KEUKEN', } , # KEUKEN 125
            'close_b': { 'label': 2743, 'folder_idx': 2787, 'sign': 'ROK', } , # ROK 678
            'far': { 'label': 907, 'folder_idx': 914, 'sign': 'DUITSLAND-A', } , # DUITSLAND-A 2158
        }, {
            'close_a': { 'label': 2326, 'folder_idx': 2365, 'sign': 'OOK-A', } , # OOK-A 2707
            'close_b': { 'label': 3663, 'folder_idx': 3713, 'sign': 'WENNEN-A', } , # WENNEN-A 3235
            'far': { 'label': 1872, 'folder_idx': 1892, 'sign': 'LEVEN-A', } , # LEVEN-A 2542
        },{
            'close_a': { 'label': 1200, 'folder_idx': 1216, 'sign': 'GEWOON-A', } , # GEWOON-A 480
            'close_b': { 'label': 3664, 'folder_idx': 3714, 'sign': 'WENNEN-A', } , # WENNEN-B 4163
            'far': { 'label': 345, 'folder_idx': 345, 'sign': 'ATTENTIE', } , # ATTENTIE 889
        }, {
            'close_a': { 'label': 1089, 'folder_idx': 1103, 'sign': 'GA-MAAR', } , # GA-MAAR 796
            'close_b': { 'label': 3339, 'folder_idx': 3387, 'sign': 'VAN-B', } , # VAN-B 819
            'far': { 'label': 3276, 'folder_idx': 3323, 'sign': 'UITDAGEN-B', } , # UITDAGEN-B 1237
        }, {
            'close_a': { 'label': 1816, 'folder_idx': 1836, 'sign': 'LAAT-MAAR', } , # LAAT-MAAR 1150
            'close_b': { 'label': 242, 'folder_idx': 242, 'sign': 'AFHANKELIJK-D', } , # AFHANKELIJK-D 2364
            'far': { 'label': 1950, 'folder_idx': 1970, 'sign': 'MAAKT-NIET-UIT', } , # MAAKT-NIET-UIT 1163
        }, {
            'close_a': { 'label': 3814, 'folder_idx': 3866, 'sign': 'ZOU', } , # ZOU 304
            'close_b': { 'label': 273, 'folder_idx': 273, 'sign': 'ALLEEN-A', } , # ALLEEN-A 375
            'far': { 'label': 622, 'folder_idx': 624, 'sign': 'BREDE-SCHOUDERS', } , # BREDE-SCHOUDERS 1616
        } , {
            'close_a': { 'label': 2847, 'folder_idx': 2891, 'sign': 'SCHOUDERKLOPJE', } , # SCHOUDERKLOPJE 2325
            'close_b': { 'label': 2846, 'folder_idx': 2890, 'sign': 'SCHOUDER', } , # SCHOUDER 2384
            'far': { 'label': 2189, 'folder_idx': 2217, 'sign': 'NIKS-A', } , # NIKS-A 2339
        }, {
            'close_a': { 'label': 2194, 'folder_idx': 2223, 'sign': 'NIKS-G', } , # NIKS-G 4164
            'close_b': { 'label': 2300, 'folder_idx': 2339, 'sign': 'ONTSPANNEN-B', } , # ONTSPANNEN-B 572
            'far': { 'label': 27, 'folder_idx': 27, 'sign': '0-A', } , # 0-A 2892
        }, {
            'close_a': { 'label': 1937, 'folder_idx': 1957, 'sign': 'LUI', } , # LUI 1840
            'close_b': { 'label': 1730, 'folder_idx': 1750, 'sign': 'KLOPT-A', } , # KLOPT-A 649
            'far': { 'label': 1847, 'folder_idx': 1867, 'sign': 'LEEGHOOFD-A', } , # LEEGHOOFD-A 2603
        }, {
            'close_a': { 'label': 223, 'folder_idx': 223, 'sign': 'ADEM-INHOUDEN-A', } , # ADEM-INHOUDEN-A 81
            'close_b': { 'label': 1990, 'folder_idx': 2010, 'sign': 'MAKKELIJK', } , # MAKKELIJK 2775
            'far': { 'label': 470, 'folder_idx': 471, 'sign': 'BENIDORM', } , # BENIDORM 3477
        }
    ])

    # Result DataFrame
    CLOSE_FAR_RESULT = pd.DataFrame(columns=[
        'Input',
        'Close_A',
        'Close_A_label',
        'Close_B',
        'Close_B_label',
        'Far',
        'Far_label',
        'Ranking_Signs',
        'Close_A_Sign',
        'Close_B_Sign',
        'Far_Sign',
    ])

    # dataset
    dataset = TSNDataSet(
        'ngt_full_4-MFFs-3f1c_10fps_op_kfe_fr2_train',
        (36, 224, 224),
        torchvision.transforms.Compose([
            # Normalize according to ImageNet means and std
            GroupNormalize(255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 36, 4, 3),
        ]),
        is_train=False, debug=False, index_label=False, compression='lzma',
    )

    # Model Config
    model_name = 'efficientnetv2'
    num_segments = 4
    num_motion = 3

    # Model
    model = TSN(
        3846, 4, 'RGBFlow', base_model=model_name, num_motion=num_motion, img_feature_dim=256, dataset='ngt',
        consensus_type='MLP', create_embeddings=False,
    )
    # model properties
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    input_rescale = model.rescale
    # Make Parallel Model
    gpus = list(range(torch.cuda.device_count()))[:1]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    cudnn.benchmark = True

    # loss function
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # load model
    config = f'MFF_ngt_full_RGBFlow_{model_name}_segment4_{num_motion}f1c'
    description = f'ngt_close_far_200_epochs_op_kfe_1'
    weights_file_path = f'model/{config}_{description}_best.pth.tar'
    if os.path.isfile(weights_file_path):
        print(f'=> Loading Checkpoint {weights_file_path}')
        checkpoint = torch.load(weights_file_path)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.module.load_state_dict(checkpoint['state_dict'])
        print(f'=> Successfully Loaded Checkpoint {weights_file_path} (epoch {start_epoch}) (best_prec1 {best_prec1:.3f})')
    else:
        raise Exception(f'=> No Checkpoint Found At {weights_file_path}')
    print('=' * 50)

    # validate(val_loader, model, criterion)
    # sys.exit()

    # put model in eval mode
    model.eval()

    # perform close far experiment
    for o in CLOSE_FAR_DICT:
        print(o)
        # Images and Labels
        close_a_img, close_a_lbl, _ = dataset.get_record_by_folder_id(o['close_a']['folder_idx'])
        close_b_img, close_b_lbl, _ = dataset.get_record_by_folder_id(o['close_b']['folder_idx'])
        far_img, far_lbl, _ = dataset.get_record_by_folder_id(o['far']['folder_idx'])
        # Sign Names
        close_a_sign = o['close_a']['sign']
        close_b_sign = o['close_b']['sign']
        far_sign = o['far']['sign']

        print(f'close_a_lbl: {close_a_lbl}, close_b_lbl: {close_b_lbl}, far_lbl: {far_lbl}')

        with torch.no_grad():
            # Input A, Compare to B and Far
            sm = torch.nn.Softmax(dim=1)
            with torch.no_grad():
                input_var = Variable(torch.tensor(close_a_img).cuda())
                output_close_a = model(input_var)
                output_close_a = sm(output_close_a).detach().cpu().numpy()[0]
            
            output_close_a_argsort = np.argsort(output_close_a)[::-1].tolist()
            print(f'output_close_a_argsort: {output_close_a_argsort[:5]}, output_close_a: {output_close_a[output_close_a_argsort[:5]]}')
            # get index or results
            a_in_close_a_idx = output_close_a_argsort.index(o['close_a']['label'])
            a_in_close_b_idx = output_close_a_argsort.index(o['close_b']['label'])
            a_in_far_idx = output_close_a_argsort.index(o['far']['label'])

            print('Close A as input index results: '\
                f'close_a={a_in_close_a_idx} ({output_close_a[a_in_close_a_idx]:.4f}), '\
                f'close_b={a_in_close_b_idx} ({output_close_a[a_in_close_b_idx]:.4f}), '\
                f'far={a_in_far_idx} ({output_close_a[a_in_far_idx]:.4f}) '\
            )

            # Save A-Input results
            CLOSE_FAR_RESULT = CLOSE_FAR_RESULT.append({
                'Input': 'A',
                'Close_A': a_in_close_a_idx, 'Close_B': a_in_close_b_idx, 'Far': a_in_far_idx,
                'Close_A_label': close_a_lbl, 'Close_B_label': close_b_lbl, 'Far_label': far_lbl,
                'Ranking_Signs': output_close_a_argsort,
                'Close_A_Sign': close_a_sign, 'Close_B_Sign': close_b_sign, 'Far_Sign': far_sign,
            }, ignore_index=True)

            # Input B, Compare to A and Far
            input_var = Variable(torch.tensor(close_b_img).cuda())
            output_close_b = model(input_var).detach().cpu().numpy()[0]
            output_close_b_argsort = np.argsort(output_close_b)[::-1].tolist()
            print(f'output_close_b_argsort: {output_close_b_argsort[:10]}')
            # get index or results
            b_in_close_a_idx = output_close_b_argsort.index(o['close_a']['label'])
            b_in_close_b_idx = output_close_b_argsort.index(o['close_b']['label'])
            b_in_far_idx = output_close_b_argsort.index(o['far']['label'])

            print('Close B as input index results: '\
                f'close_a={b_in_close_a_idx} ({output_close_a[b_in_close_a_idx]:.4f}), '\
                f'close_b={b_in_close_b_idx} ({output_close_a[b_in_close_b_idx]:.4f}), '\
                f'far={b_in_far_idx} ({output_close_a[b_in_far_idx]:.4f}) '\
            )

            # Save B-Input Results
            CLOSE_FAR_RESULT = CLOSE_FAR_RESULT.append({
                'Input': 'B',
                'Close_A': b_in_close_a_idx, 'Close_B': b_in_close_b_idx, 'Far': b_in_far_idx,
                'Close_A_label': close_a_lbl, 'Close_B_label': close_b_lbl, 'Far_label': far_lbl,
                'Ranking_Signs': output_close_b_argsort,
                'Close_A_Sign': close_a_sign, 'Close_B_Sign': close_b_sign, 'Far_Sign': far_sign,
            }, ignore_index=True)

            print(f'=' * 50)

    # Save all results as Excel file
    file_path_out = f'output/{config}_{description}_{datetime.now().strftime("%d-%m-%Y_%I%p")}.xlsx'
    print(f'Writing output to {file_path_out}')
    CLOSE_FAR_RESULT.to_excel(file_path_out, index=False)
    print(CLOSE_FAR_RESULT)

def validate(val_loader, model, criterion, epoch=0):
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
    for i, (imgs, lbls) in enumerate(tqdm(val_loader)):
        # a = np.moveaxis(imgs.numpy()[0, 6:9], 0, 2)
        # print(a.shape)
        # a = (a * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        # a = (a*255).astype(np.uint8)
        # Image.fromarray(a).show()
        # sys.exit()
        # if i == 100: break

        lbls = lbls.cuda()
        with torch.no_grad():
            input_var = Variable(imgs)
            target_var = Variable(lbls)
            # compute output
            output = model(input_var)

        loss = criterion(output, target_var)

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
    output = ('Val   Epoch: [{}][{}/{}], \t'
                'Time {:.3f}\t Data {:.3f}\t Loss {:.4f}\t'
                'Prec@1 {:.3f}\t Prec@5 {:.3f} \t Prec@100 {:.3f} \t Prec@1000 {:.3f}'
            .format(
                    epoch+1, i+1, len(val_loader), 
                    batch_time.avg, data_time.avg, losses.avg, top1.avg, top5.avg, top100.avg, top1000.avg
                )
            )
            
    print(output)

    if epoch == 0:
        imgs_stats = imgs.shape, imgs.mean(), imgs.std(), imgs.min(), imgs.max(), imgs.dtype, type(imgs)
        print('imgs shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % imgs_stats)
        lbls_stats = lbls.shape, lbls.float().mean(), lbls.float().std(), lbls.min(), lbls.max(), lbls.dtype, type(lbls)
        print('lbls shape: %s, mean: %.2f, std: %.2f, min: %.2f, max: %.2f, dtype: %s, type: %s' % lbls_stats)

    return top1.avg

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

if __name__ == '__main__':
    main()