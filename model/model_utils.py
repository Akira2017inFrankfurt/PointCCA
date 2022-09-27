import copy
import torch
import math
import torch.nn as nn


@torch.no_grad()
def momentum_update(online, target, tao=0.99):
    """
    :param online: online encoder
    :param target: target encoder to update
    :param tao: update parameter
    :return: updated target encoder
    """
    if target is None:
        target = copy.deepcopy(online)
    else:
        for online_params, target_params in zip(online.parameters(), target.parameters()):
            target_weight, online_weight = target_params.data, online_params.data
            target_params.data = target_weight * tao + (1 - tao) * online_weight
    for parameter in target.parameters():
        parameter.requires_grad = False
    return target


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if not dim9:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    local_device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=local_device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 2*num_dims, num_points, k)


def get_patches_feature(patches, patch_encoder):
    """
    :param patches: [B, 8, n, f]
    :param patch_encoder: input->[B, n, f] +++ output->[B, 1024]
    :return: [B, 8, 1024]
    """
    num_patches = patches.shape[1]
    patch_features = patch_encoder(torch.squeeze(patches[:, 0, :, :]))
    for i in range(1, num_patches):
        current_patch_feature = patch_encoder(torch.squeeze(patches[:, i, :, :]))
        patch_features = torch.cat((patch_features, current_patch_feature), dim=1)
    return patch_features


def loss_fn(x, y):
    """
    contrastive loss for classification task
    """
    x = torch.nn.functional.normalize(x, dim=-1, p=2)
    y = torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def seg_loss_fn(x, y, proj):
    """
    contrastive loss for part_ and sem_segmentation
    """
    l_loss = 0.0
    for i in range(x.shape[2]):
        l_loss += loss_fn(proj(x[:, :, i]), y[:, :, i])
    return l_loss / x.shape[2]


def sample_and_group(npoint, nsample, xyz, points):
    def square_distance(src, dst):
        return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

    def index_points(points, idx):
        raw_size = idx.size()
        idx = idx.reshape(raw_size[0], -1)
        res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
        return res.reshape(*raw_size, -1)

    def farthest_point_sample(xyz, npoint):
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint)  # [B, n_point]

    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    dists = square_distance(new_xyz, xyz)  # B x n_point x N
    idx = dists.argsort()[:, :, :nsample]  # B x n_point x K

    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points


class SA_MH_Layer(nn.Module):
    """
    self-attention layer with multi-head-attention
    """
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, f, p = x.shape
        x_q = self.q_conv(x)
        x_q = x_q.reshape(bs, 4, -1, p).permute(0, 1, 3, 2)
        x_k = self.k_conv(x)
        x_k = x_k.reshape(bs, 4, -1, p)
        xy = torch.matmul(x_q, x_k)
        xy = self.softmax(xy / math.sqrt(x_k.shape[-2]))

        x_v = self.v_conv(x)
        x_v = x_v.reshape(bs, 4, -1, p).permute(0, 1, 3, 2)
        xyz = torch.matmul(xy, x_v)
        xyz = xyz.permute(0, 1, 3, 2).reshape(bs, -1, p)
        xyz = self.act(self.after_norm(self.trans_conv(xyz - x)))
        xyz = x + xyz
        return xyz


class CrossAttention(nn.Module):
    """
    cross attention fo classification task
    """
    def __init__(self, channel=1024):
        super().__init__()
        self.q_conv = nn.Conv1d(channel, channel // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channel, channel // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(channel, channel, 1, bias=False)
        self.trans_conv = nn.Conv1d(channel, channel, 1)
        self.after_norm = nn.BatchNorm1d(channel)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_tensor, kv_tensor):
        x_q = self.q_conv(q_tensor.permute(0, 2, 1))
        x_k = self.k_conv(kv_tensor.permute(0, 2, 1))
        x_v = self.v_conv(kv_tensor.permute(0, 2, 1))

        energy = torch.matmul(x_q.permute(0, 2, 1), x_k)
        attention = self.softmax(energy / math.sqrt(x_k.shape[-2]))
        x_r = torch.matmul(attention, x_v.permute(0, 2, 1))
        res = (q_tensor - x_r).permute(0, 2, 1)
        x_r = self.act(self.after_norm(self.trans_conv(res)))
        x_r = x_r.permute(0, 2, 1) + q_tensor

        return x_r


class SegXAttn(nn.Module):
    """
    cross attention for segmentation task
    """
    def __init__(self, q_in=256, q_out=256, k_in=1024):
        super().__init__()
        self.q_in = q_in
        self.q_out = q_out
        self.k_in = k_in

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, q_out, 1, bias=False)
        self.v_conv = nn.Conv1d(k_in, q_in, 1, bias=False)

        self.trans_conv = nn.Conv1d(q_in, q_in, 1)
        self.after_norm = nn.BatchNorm1d(q_in)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q_tensor, kv_tensor):
        # print('q_tensor: ', q_tensor.shape)
        # N, 256 ---> N, 256
        x_q = self.q_conv(q_tensor)
        # print('x_q: ', x_q.shape)
        # 16, 1024 ---> 16, 256
        x_k = self.k_conv(kv_tensor.permute(0, 2, 1))
        # print('x_k: ', x_k.shape)
        # 16, 1024 ---> 16, 256
        x_v = self.v_conv(kv_tensor.permute(0, 2, 1))
        # print('x_v: ', x_v.shape)
        # N, 16
        energy = torch.matmul(x_q.permute(0, 2, 1), x_k)
        # print('energy: ', energy.shape)
        attention = self.softmax(energy / math.sqrt(x_k.shape[-2]))
        x_r = torch.matmul(attention, x_v.permute(0, 2, 1))
        # print('x_r shape: ', x_r.shape)
        res = (q_tensor - x_r.permute(0, 2, 1))
        # print('res: ', res.shape)
        x_r = self.act(self.after_norm(self.trans_conv(res)))
        # print('last x_r:', x_r.shape)
        x_r = x_r + q_tensor

        return x_r


class Projector(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, hidden_size=4096):
        super(Projector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = self.bn(self.l1(x.reshape(x.shape[0], -1)))
        x = self.l2(self.relu(x))
        return x.reshape(x.shape[0], 1, -1)


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print('original x shape: ', x.shape)
        x = x.permute(0, 3, 1, 2)
        # print('after permute x shape: ', x.shape)
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = torch.max(x, 3)[0]
        return x


class StackedAttention(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_MH_Layer(channels)
        self.sa2 = SA_MH_Layer(channels)
        self.sa3 = SA_MH_Layer(channels)
        self.sa4 = SA_MH_Layer(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x