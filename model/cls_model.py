import torch
import copy
import torch.nn as nn
from model_utils import * 


class Encoder_Head(nn.Module):
    def __init__(self):
        super().__init__()
        d_points = 3
        self.conv1 = nn.Conv1d(d_points, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)
        self.pt_last = StackedAttention()

        self.relu = nn.ReLU()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        xyz = x[..., 0:3]
        x = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, nsample=32, xyz=xyz, points=x)
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, nsample=32, xyz=new_xyz, points=feature)
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        return x.reshape(batch_size, 1, -1)



class CLS_Model(nn.Module):
    def __init__(self):
        super(CLS_Model, self).__init__()
        self.task_type = 'cls'
        self.down_sample_func = fps_downsample
        self.divide_patches_func = get_patches
        self.online_encoder = Encoder_Head().to(device)
        
        self.online_cross_attention = CrossAttention().to(device)
        self.get_projection = Projector().to(device)
        self.target_encoder = None
        self.target_cross_attention = None

        self.net = Base_Structure(
            task_type=self.task_type,
            get_down_sample=self.down_sample_func,
            get_patches=self.divide_patches_func,
            get_encoder_point=self.online_encoder,
            get_encoder_patch = None,
            get_cross_attention=self.online_cross_attention,
            get_projection=self.get_projection
        )

    def forward(self, aug1, aug2):
        return self.net(aug1, aug2)
