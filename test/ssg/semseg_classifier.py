import torch
import torch.nn as nn
from sem_seg.sem_model import Encoder_Head_Sem


def get_encoder(model_weights_path):
    loaded_paras = torch.load(model_weights_path)
    # encoder = Encoder_Head_Sem().to(device)
    encoder = Encoder_Head_Sem().cuda()
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
    encoder = Encoder_Head_Sem().cuda()
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


class SemSegClassifier_1(nn.Module):
    # in->256
    # 256->256, 256->128, 128->64, 64->13
    def __init__(self, model_weights_save_path):
        super(SemSegClassifier_1, self).__init__()
        self.encoder = get_encoder(model_weights_path=model_weights_save_path)

        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(64, 13, kernel_size=1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        N = x.shape[1]

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


class SemSegClassifier_2(nn.Module):
    # in->256
    # 256->512, 512->256, 256->128, 128->13
    def __init__(self, model_weights_save_path):
        super(SemSegClassifier_2, self).__init__()
        self.encoder = get_encoder(model_weights_path=model_weights_save_path)

        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)

        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, 13, kernel_size=1, bias=False)

    def forward(self, x):
        B = x.shape[0]
        N = x.shape[1]

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


class SemSegClassifier_3(nn.Module):
    # in->256
    # 256->256, 256->128, 128->13
    def __init__(self, model_weights_save_path):
        super(SemSegClassifier_3, self).__init__()
        self.encoder = get_encoder(model_weights_path=model_weights_save_path)

        self.bn0 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)

        self.conv2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)

        # self.conv3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(128),
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Conv1d(128, 13, kernel_size=1, bias=False)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)  # B, 256, N
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.dp1(x)
        x = self.conv2(x)
        x = self.dp2(x)
        # x = self.conv3(x)
        x = self.conv4(x)
        return x


class SemSegClassifier_4(nn.Module):
    # in->256 + 1024
    # 256->256, 256->128, 128->13
    def __init__(self, model_weights_save_path):
        super(SemSegClassifier_4, self).__init__()
        self.encoder, self.tail = init_encoder(model_weights_save_path)

        self.bn0 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv1d(256, 13, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(128),
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)  # B, 256, N
        x = self.bn0(x)  # point-wise features
        num_points = x.shape[-1]
        global_feature = self.tail(x)  # B, 1024, 1
        global_feature = global_feature.repeat(1, 1, num_points)

        x = torch.cat((x, global_feature), dim=1)  # B, 1280, N
        x = self.conv1(x)  # 1280->512
        x = self.conv2(x)  # 512 -> 256
        x = self.dp1(x)
        x = self.conv3(x)  # 256 -> 13
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = r'/home/haruki/下载/SimAttention/sem_seg/semseg_test/weights/model_sem_seg_v1-23.pth'
    # loaded_paras = torch.load(model_path)
    # for k in loaded_paras.keys():
    #     print(k)
    # count_po = 0  # 95
    # count_pa = 0  # 101
    # # for k in loaded_paras.keys():
    # #     if k.startswith('online_encoder'):
    # #         count_po += 1
    # #     if k.startswith('patch_encoder'):
    # #         count_pa += 1
    # # print(count_po, count_pa)
    # from sem_seg.sem_model import Encoder_Patch_Sem
    #
    # en_1 = Encoder_Head_Sem()
    # en_2 = Encoder_Patch_Sem(en_1)
    # en_1_dict = en_1.state_dict()
    # en_2_dict = en_2.state_dict()
    # print('point wise encoder: ', len(en_1_dict.keys()))  # 95
    # print('patch wise encoder: ', len(en_2_dict.keys()))  # 101

    # en, tail = init_encoder(model_path)
    rand_x_1 = torch.rand([2, 4096, 6]).to(device)
    model = SemSegClassifier_4(model_path).to(device)
    output = model(rand_x_1)
