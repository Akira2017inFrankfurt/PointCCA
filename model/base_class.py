"""
 - Description: Basic pipeline structure
 - Author: Qianliang Huang
 - Date: 2022.09.23
"""

import torch.nn as nn
from model_utils import *


class Base_Structure(nn.Module):
    def __init__(self,
                 task_type,
                 get_down_sample,
                 get_patches,
                 get_encoder_point,
                 get_encoder_patch,
                 get_cross_attention,
                 get_projection
                 ):
        super().__init__()
        self.task_type = task_type  # 'cls', 'part_seg', sem_seg'
        self.down_sample_func = get_down_sample
        self.divide_patches_func = get_patches
        self.online_encoder = get_encoder_point
        self.patch_encoder = get_encoder_patch
        self.online_cross_attention = get_cross_attention
        self.get_projection = get_projection
        self.target_encoder = None
        self.target_cross_attention = None

    def forward(self, aug1, aug2):
        # set target branch encoder and cross attention
        self.target_encoder = momentum_update(self.online_encoder, self.target_encoder)
        self.target_cross_attention = momentum_update(self.online_cross_attention, self.target_cross_attention)

        # get down samples from 2 augmented point clouds
        if self.task_type != 'cls':
            down_sample1, down_sample2 = aug1, aug2
        else:
            _, down_sample1 = self.down_sample_func(aug1, 1024)
            _, down_sample2 = self.down_sample_func(aug2, 1024)

        # get patches
        patches_1 = self.divide_patches_func(down_sample1)
        patches_2 = self.divide_patches_func(down_sample2)

        # get point features/global features
        point_feature_1 = self.online_encoder(down_sample1)
        point_feature_2 = self.target_encoder(down_sample2)
        point_feature_3 = self.online_encoder(down_sample2)
        point_feature_4 = self.target_encoder(down_sample1)

        # get patch features
        patch_feature_1 = get_patches_feature(patches_1, self.patch_encoder)
        patch_feature_2 = get_patches_feature(patches_2, self.patch_encoder)
        patch_features = torch.cat((patch_feature_1, patch_feature_2), dim=1)

        # get cross attention
        cross_attention_1 = self.online_cross_attention(point_feature_1, patch_features)
        cross_attention_2 = self.target_cross_attention(point_feature_2, patch_features)
        cross_attention_3 = self.online_cross_attention(point_feature_3, patch_features)
        cross_attention_4 = self.target_cross_attention(point_feature_4, patch_features)

        # get contrastive loss
        loss_func = loss_fn if self.task_type == 'cls' else seg_loss_fn
        loss_1 = loss_func(cross_attention_1, cross_attention_2, self.get_projection)
        loss_2 = loss_func(cross_attention_3, cross_attention_4, self.get_projection)

        return loss_1 + loss_2
