import torch
import torch.nn as nn
from model_utils import SA_MH_Layer, get_graph_feature, SegXAttn, Projector
from base_class import Base_Structure
from data.data_sample_utils import fps_downsample, get_patches

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder_Head(nn.Module):
    def __init__(self, k=40, part_num=50):
        super(Encoder_Head, self).__init__()
        self.k = k
        self.part_num = part_num
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.sa1 = SA_MH_Layer(128)
        self.sa2 = SA_MH_Layer(128)
        self.sa3 = SA_MH_Layer(128)
        self.sa4 = SA_MH_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B,N,3] -> [B,3,N]
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64*2, num_points)
        # b, 128, 2048
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        # 128 * 4 = 512, [B, 512, N]
        x = torch.concat((x1, x2, x3, x4), dim=1)
        # 512 --> 256, [B, 256, N]
        x = self.conv_fuse(x)
        # x_max = torch.max(x, 2)[0]  # [B, 1024]
        return x


class Encoder_Patch(nn.Module):
    def __init__(self, encoder_head):
        super(Encoder_Patch, self).__init__()
        self.encoder_head = encoder_head
        self.conv_fuse = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.shape[0]
        pointwise_feature = self.encoder_head(x)
        pa = self.conv_fuse(pointwise_feature)
        x = torch.max(pa, 2)[0]
        patch_feature = x.view(batch_size, -1)
        return patch_feature.reshape(batch_size, 1, -1)


class PartSegModel(nn.Module):
    def __init__(self):
        super(PartSegModel, self).__init__()
        self.task_type = 'part_seg'
        self.down_sample_func = fps_downsample
        self.divide_patches_func = get_patches
        self.online_encoder = Encoder_Head().to(device)
        self.patch_encoder = Encoder_Patch(self.online_encoder).to(device)
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

