from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
from torchinfo import summary
import pandas as pd

import torch.nn.functional as F
import pretrainedmodels
import MLPmodule
import geffnet
import timm 

class TSN(nn.Module):
    def __init__(self, args, base_model='resnet101', new_length=None,
                 before_softmax=True, num_motion=3, dropout=0.8, dataset='jester',
                 crop_num=1, partial_bn=True, print_spec=True, softmax=False, 
            ):
        super(TSN, self).__init__()
        self.name = base_model
        self.num_class = args.num_class
        self.modality = args.modality
        self.num_segments = args.num_segments
        self.num_motion = num_motion
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.dataset = args.dataset
        self.crop_num = crop_num
        self.consensus_type = args.consensus_type
        self.img_feature_dim = args.img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.multi_sim_label = args.multi_sim_label
        self.create_embeddings = args.create_embeddings
        self.arcface_head = args.arcface_head
        if self.arcface_head:
            self.arcmargin = ArcMarginProduct(512, self.num_class, s=args.arcface_s, m=args.arcface_m, ls_eps=arcface_ls)

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            if self.modality == "RGB":
                self.new_length = 1
            elif self.modality == "Flow":
                self.new_length = 5
            elif self.modality == "RGBFlow":
                #self.new_length = 1
                self.new_length = self.num_motion 
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, self.consensus_type, self.dropout, self.img_feature_dim)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn()

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        elif self.modality == 'RGBFlow':
            print("Converting the ImageNet model to RGB+Flow init model")
            self.base_model = self._construct_rgbflow_model(self.base_model)
            print("Done. RGBFlow model ready.")
        
        if self.consensus_type == 'MLP':
            self.consensus = MLPmodule.return_MLP(
                    self.img_feature_dim, self.num_segments, self.num_class, args.dropout_extra,
                    args.multi_sim_label, args.create_embeddings, self.arcface_head,
                )
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)


    def _prepare_tsn(self):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        # if self.dropout == 0:
        #     setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, self.num_class))
        #     self.new_fc = None
        # else:
        setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout, inplace=True))
        if self.consensus_type == 'MLP':
            # set the MFFs feature dimension
            self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
        else:
            # the default consensus types in TSN
            self.new_fc = nn.Linear(feature_dim, self.num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model or 'squeezenet1_1' in base_model:
            self.base_model = pretrainedmodels.__dict__[base_model](num_classes=1000, pretrained='imagenet')
            if base_model == 'squeezenet1_1':
                self.base_model = self.base_model.features
                self.base_model.last_layer_name = '12'
            else:
                self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        elif base_model == 'BNInception':
            self.base_model = pretrainedmodels.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.rescale = 255
        elif 'resnext101' == base_model:
            self.base_model = pretrainedmodels.__dict__[base_model](num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
        elif 'efficientnet' == base_model:
            self.base_model = geffnet.create_model('tf_efficientnet_b3_ns', pretrained=True)
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.rescale = 255
        elif 'efficientnetv2_s' == base_model:
            self.base_model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.rescale = 255
        elif base_model.startswith('tf_efficientnetv2_b'):
            self.base_model = timm.create_model(base_model, pretrained=True)
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.rescale = 255
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
            
        summary(self.base_model, input_size=(1, 3, 224, 224))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            # print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False


    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear) or isinstance(m, ArcMarginProduct):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, labels):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        if self.modality == 'RGBFlow':
            sample_len = 3 + 2 * self.new_length

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        # if self.dropout > 0:
        base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        # print(f'Consensus out: {pd.Series(output.cpu().detach().numpy().flatten()).describe()}')
        # print(f'consensus out dtype: {output.dtype}')

        # ArcMargin Head
        if self.arcface_head:
            output = self.arcmargin(output, labels)
        # print(f'arcface_head out: {pd.Series(output.cpu().detach().numpy().flatten()).describe()}')
        # print(f'arcface_head out dtype: {output.dtype}')

        if type(output) is not tuple:
            return output.squeeze(1)
        else:
            output_squeeze = []
            for o in output:
                output_squeeze.append(o.squeeze(1))
            return output_squeeze

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    """ # There is no need now!!
    def _get_rgbflow(self, input):
        input_c = 3 + 2 * self.new_length # 3 is rgb channels, and 2 is coming for x & y channels of opt.flow
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        new_data = input_view.clone()
        return new_data
    """

    def _construct_rgbflow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        filter_conv2d = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))
        first_conv_idx = next(filter_conv2d)
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
      
        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = torch.cat((params[0].data.mean(dim=1,keepdim=True).expand(new_kernel_size).contiguous(), params[0].data), 1) # NOTE: Concatanating might be other way around. Check it!
        new_kernel_size = kernel_size[:1] + (3 + 2 * self.new_length,) + kernel_size[2:]
        
        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name
        
        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        print(kernel_size)
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        print(new_kernel_size)
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

# source: https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
class ArcMarginProduct(torch.nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s, m, easy_margin=False, ls_eps=0.0):
        print(f'Initialising ArcMarginProduct with s: {s}, m: {m}, ls: {ls_eps}')
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
