import torch
import torch.nn as nn
from model_utils import SA_MH_Layer, get_graph_feature, SegXAttn, Projector
from base_class import Base_Structure
from data.data_sample_utils import fps_downsample, get_patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder_Head_Sem(nn.Module):
    def __init__(self):
        super(Encoder_Head_Sem, self).__init__()
        self.k = 20

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.sa1 = SA_MH_Layer(128)
        self.sa2 = SA_MH_Layer(128)
        self.sa3 = SA_MH_Layer(128)
        self.sa4 = SA_MH_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        # x [B, N, f]---> [B, 9, 4096]
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k, dim9=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)

        x1 = self.sa1(x)
        x2 = self.sa1(x1)
        x3 = self.sa1(x2)
        x4 = self.sa1(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        return x


class Encoder_Patch_Sem(nn.Module):
    def __init__(self, encoder_head_sem):
        super(Encoder_Patch_Sem, self).__init__()
        self.encoder_head = encoder_head_sem
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pointwise_feature = self.encoder_head(x)
        patch_feature = self.conv_fuse(pointwise_feature)
        patch_feature = torch.max(patch_feature, 2)[0]
        patch_feature = patch_feature.view(batch_size, -1)
        return patch_feature.reshape(batch_size, 1, -1)


class SemSegModel(nn.Module):
    def __init__(self):
        super(SemSegModel, self).__init__()
        self.task_type = 'sem_seg'
        self.down_sample_func = fps_downsample
        self.divide_patches_func = get_patches
        self.online_encoder = Encoder_Head_Sem().to(device)
        self.patch_encoder = Encoder_Patch_Sem(self.online_encoder).to(device)
        self.online_cross_attention = SegXAttn().to(device)
        self.get_projection = Projector().to(device)
        self.target_encoder = None
        self.target_cross_attention = None

        self.net = Base_Structure(
            task_type=self.task_type,
            get_down_sample=self.down_sample_func,
            get_patches=self.divide_patches_func,
            get_encoder_point=self.online_encoder,
            get_encoder_patch=self.patch_encoder,
            get_cross_attention=self.online_cross_attention,
            get_projection=self.get_projection
        )

    def forward(self, aug1, aug2):
        return self.net(aug1, aug2)

