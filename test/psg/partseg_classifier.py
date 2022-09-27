import torch
import torch.nn as nn
from psg_model_3 import Encoder_Head
from partseg_hf5 import ShapeNetPart_Original
from torch.utils.data import DataLoader


def get_encoder(model_weights_path):
    loaded_paras = torch.load(model_weights_path)
    # encoder = Encoder_Head().to(device)
    encoder = Encoder_Head().cuda()
    encoder_dict = encoder.state_dict()
    new_state_dict = {}

    for k in loaded_paras.keys():
        if k.startswith('online_encoder'):
            new_k = k[15:]
            new_state_dict[new_k] = loaded_paras[k]

    encoder_dict.update(new_state_dict)
    encoder.load_state_dict(encoder_dict)
    return encoder


def init_encoder(weight_path):
    loaded_paras = torch.load(weight_path)
    encoder = Encoder_Head().cuda()
    encoder_dict = encoder.state_dict()
    new_state_dict = {}

    tail = Encoder_Tail().cuda()
    tail_dict = tail.state_dict()
    new_tail_dict = {}

    for k in loaded_paras.keys():
        if k.startswith('online_encoder'):
            new_k = k[15:]
            new_state_dict[new_k] = loaded_paras[k]
        if k.startswith('patch_encoder.conv_fuse'):
            new_k = k[14:]
            new_tail_dict[new_k] = loaded_paras[k]

    encoder_dict.update(new_state_dict)
    encoder.load_state_dict(encoder_dict)
    tail_dict.update(new_tail_dict)
    tail.load_state_dict(tail_dict)
    return encoder, tail


class Encoder_Tail(nn.Module):
    def __init__(self):
        super(Encoder_Tail, self).__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        patch_feature = self.conv_fuse(x)
        patch_feature = torch.max(patch_feature, 2)[0]
        patch_feature = patch_feature.view(batch_size, -1)
        return patch_feature.reshape(batch_size, -1, 1)


class PartSegClassifier(nn.Module):
    def __init__(self, model_weights_save_path):
        super(PartSegClassifier, self).__init__()
        self.encoder = get_encoder(model_weights_path=model_weights_save_path)
        self.conv_label = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(320, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

    def forward(self, x, one_hot_label):
        B = x.shape[0]
        N = x.shape[1]

        one_hot_label = one_hot_label.view(B, -1, 1)
        one_hot_label = self.conv_label(one_hot_label)
        one_hot_label = one_hot_label.repeat(1, 1, N)

        with torch.no_grad():
            x = self.encoder(x)  # B, 256, N
        x = self.bn0(x)
        x = torch.cat((x, one_hot_label), dim=1)
        x = self.conv1(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = self.dp2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class PartSegClassifier_2(nn.Module):
    def __init__(self, model_weights_save_path):
        super(PartSegClassifier_2, self).__init__()
        self.encoder, self.tail = init_encoder(model_weights_save_path)
        self.conv_label = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(320+1024, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, 50, kernel_size=1, bias=False)

    def forward(self, x, one_hot_label):
        B = x.shape[0]
        N = x.shape[1]

        one_hot_label = one_hot_label.view(B, -1, 1)
        one_hot_label = self.conv_label(one_hot_label)
        one_hot_label = one_hot_label.repeat(1, 1, N)

        with torch.no_grad():
            x = self.encoder(x)  # B, 256, N
        x = self.bn0(x)

        global_feature = self.tail(x)  # B, 1024, 1
        global_feature = global_feature.repeat(1, 1, N)
        x = torch.cat((x, global_feature, one_hot_label), dim=1)

        x = self.conv1(x)  # 1344 -> 256
        x = self.dp1(x)
        x = self.conv2(x)  # 256 -> 256
        x = self.dp2(x)
        x = self.conv3(x)  # 256 -> 128
        x = self.conv4(x)  # 128 -> 50
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_name = 'model_seg_partseg_3_98.pth'
    # model_save_path = os.path.join('/home/haruki/下载/SimAttention/scripts/psg_weights', model_name)
    # rand_x_1 = torch.rand([4, 2048, 3]).to(device)
    # rand_l_1 = torch.rand([4, 16]).to(device)
    # psc = PartSegClassifier(model_save_path).to(device)
    # result = psc(rand_x_1, rand_l_1)
    # print(result)

    # test for dataloader
    root = r'/home/haruki/下载/shapenet/shapenet_part_seg_hdf5_data'
    train_dataset = ShapeNetPart_Original(root=root, num_points=2048, class_choice=None)
    # print(len(train_dataset)) # 12137
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=2, shuffle=True, drop_last=True)
    for data, label, seg, hot in train_loader:
        # print(data.shape)  # torch.Size([2, 2048, 3])
        # print(label.shape)  # torch.Size([2, 1])
        # print(seg.shape)  # torch.Size([2, 2048])
        # print(hot.shape)  # torch.Size([2, 16])
        break
