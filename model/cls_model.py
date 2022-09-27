import torch
import copy
import torch.nn as nn
from model_utils import loss_fn, Local_op, StackedAttention, sample_and_group


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
    def __init__(self,
                 sub_function,
                 knn_function,
                 online_encoder,
                 project_method,
                 crossed_attention_method):
        super().__init__()
        self.sub_function = sub_function
        self.knn_function = knn_function

        self.online_encoder = online_encoder
        self.target_encoder = None

        self.online_x_attn = crossed_attention_method
        self.target_x_attn = None

        self.projector = project_method

    def forward(self, aug1, aug2):
        # B, 1024, 3
        _, sub1 = self.sub_function(aug1, 1024)
        _, sub2 = self.sub_function(aug2, 1024)

        # B, 8, 1024, 3
        knn_patch1 = self.knn_function(aug1)
        knn_patch2 = self.knn_function(aug2)

        # [B, 1, N_f] N_f: 1024
        sub_feature_1 = self.online_encoder(sub1)
        sub_feature_3 = self.online_encoder(sub2)

        # with momentum encoder
        with torch.no_grad():
            if self.target_encoder is None:
                self.target_encoder = copy.deepcopy(self.online_encoder)
            else:
                for online_params, target_params in zip(self.online_encoder.parameters(),
                                                        self.target_encoder.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            sub_feature_2 = self.target_encoder(sub2)
            sub_feature_4 = self.target_encoder(sub1)

        # cube feature  [B, 1, N_f]
        knn_feature_1_1 = self.online_encoder(knn_patch1[:, 0, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_2 = self.online_encoder(knn_patch1[:, 1, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_3 = self.online_encoder(knn_patch1[:, 2, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_4 = self.online_encoder(knn_patch1[:, 3, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_5 = self.online_encoder(knn_patch1[:, 4, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_6 = self.online_encoder(knn_patch1[:, 5, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_7 = self.online_encoder(knn_patch1[:, 6, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_1_8 = self.online_encoder(knn_patch1[:, 7, :, :].reshape(-1, 1024, 3).contiguous())

        knn_feature_2_1 = self.online_encoder(knn_patch2[:, 0, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_2 = self.online_encoder(knn_patch2[:, 1, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_3 = self.online_encoder(knn_patch2[:, 2, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_4 = self.online_encoder(knn_patch2[:, 3, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_5 = self.online_encoder(knn_patch2[:, 4, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_6 = self.online_encoder(knn_patch2[:, 5, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_7 = self.online_encoder(knn_patch2[:, 6, :, :].reshape(-1, 1024, 3).contiguous())
        knn_feature_2_8 = self.online_encoder(knn_patch2[:, 7, :, :].reshape(-1, 1024, 3).contiguous())

        # crop feature concat [B, 8, N_f]
        crop_feature_1 = torch.cat((knn_feature_1_1, knn_feature_1_2,
                                    knn_feature_1_3, knn_feature_1_4,
                                    knn_feature_1_5, knn_feature_1_6,
                                    knn_feature_1_7, knn_feature_1_8,), dim=1)

        crop_feature_2 = torch.cat((knn_feature_2_1, knn_feature_2_2,
                                    knn_feature_2_3, knn_feature_2_4,
                                    knn_feature_2_5, knn_feature_2_6,
                                    knn_feature_2_7, knn_feature_2_8,), dim=1)
        # [B, 16, N_f]
        crop_feature = torch.cat((crop_feature_1, crop_feature_2), dim=1)

        # momentum attention feature
        with torch.no_grad():
            if self.target_x_attn is None:
                self.target_x_attn = copy.deepcopy(self.online_x_attn)
            else:
                for online_params, target_params in zip(self.online_x_attn.parameters(),
                                                        self.target_x_attn.parameters()):
                    target_weight, online_weight = target_params.data, online_params.data
                    # moving average decay is tao
                    tao = 0.99
                    target_params.data = target_weight * tao + (1 - tao) * online_weight
            for parameter in self.target_x_attn.parameters():
                parameter.requires_grad = False
            attn_feature_2 = self.target_x_attn(sub_feature_2, crop_feature)
            attn_feature_4 = self.target_x_attn(sub_feature_4, crop_feature)

        # online attention feature
        attn_feature_1 = self.online_x_attn(sub_feature_1, crop_feature)
        attn_feature_3 = self.online_x_attn(sub_feature_3, crop_feature)

        # loss
        loss_1 = loss_fn(self.project_method(attn_feature_1), attn_feature_2)
        loss_2 = loss_fn(self.project_method(attn_feature_3), attn_feature_4)
        loss = loss_1 + loss_2

        return loss.mean()
