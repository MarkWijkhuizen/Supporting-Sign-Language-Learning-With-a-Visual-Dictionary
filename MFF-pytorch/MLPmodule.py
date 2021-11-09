import torch
import math
import torch.nn as nn

class MLPmodule(torch.nn.Module):
    """
    This is the 2-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """
    def __init__(self, img_feature_dim, num_frames, num_class, dropout_extra,
        multi_sim_label, create_embeddings, arcface_head,
    ):
        super(MLPmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = 512
        self.multi_sim_label = multi_sim_label
        self.create_embeddings = create_embeddings
        self.arcface_head = arcface_head

        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(self.num_frames * self.img_feature_dim, self.num_bottleneck)
        self.do1 = nn.Dropout(dropout_extra, inplace=True)
        
        if self.multi_sim_label:
            self.relu2 = nn.ReLU()
            self.out1 = nn.Linear(self.num_bottleneck, self.num_class)
            self.out2 = nn.Linear(self.num_bottleneck, self.num_class)
            self.out3 = nn.Linear(self.num_bottleneck, self.num_class)
            self.out4 = nn.Linear(self.num_bottleneck, self.num_class)
            self.out5 = nn.Linear(self.num_bottleneck, self.num_class)
        elif self.arcface_head:
            self.relu2 = nn.Identity()
            self.out1 = nn.BatchNorm1d(self.num_bottleneck)
        else:
            self.relu2 = nn.ReLU()
            self.out1 = nn.Linear(self.num_bottleneck, self.num_class)
        
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        
        x = self.relu1(input)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.relu2(x)

        if self.multi_sim_label:
            out1 = self.out1(x).float()
            out2 = self.out2(x).float()
            out3 = self.out3(x).float()
            out4 = self.out4(x).float()
            out5 = self.out5(x).float()
            return out1, out2, out3, out4, out5
        else:
            if self.create_embeddings:
                return x.float()
            else:
                return self.out1(x).float()

def return_MLP(img_feature_dim, num_frames, num_class, dropout_extra, multi_sim_label, create_embeddings, arcface_head):
    return MLPmodule(img_feature_dim, num_frames, num_class, dropout_extra, multi_sim_label, create_embeddings, arcface_head)
