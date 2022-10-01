import os
import glob
import json
import h5py
import random
import numpy as np
from torch.utils.data import Dataset
from data.augmentation import PointWOLF


def load_data_partseg(root, partition, scale=1.0):
    data_dir = root
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(data_dir, 'data', '*train*.h5')) \
               + glob.glob(os.path.join(data_dir, 'data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(data_dir, 'data', '*%s*.h5' % partition))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        local_label = f['label'][:].astype('int64')
        local_seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(local_label)
        all_seg.append(local_seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    if partition == 'trainval':
        index_list = [i for i in range(len(all_data))]
        random.shuffle(index_list)
        scale_length = int(scale * len(all_data)) + 1
        scale_index = np.array(index_list[:scale_length])
        all_data = all_data[scale_index]
        all_label = all_label[scale_index]
        all_seg = all_seg[scale_index]
    return all_data, all_label, all_seg


def get_label_1_hot(local_label):
    label_one_hot = np.zeros((local_label.shape[0], 16))
    for idx in range(local_label.shape[0]):
        label_one_hot[idx, local_label[idx]] = 1
    label_one_hot = label_one_hot.astype(np.float32)
    return label_one_hot


class ShapeNetPart_Original(Dataset):
    # modified version, for label transformed to [B, 16] one hot vector
    def __init__(self, root, num_points, partition='trainval', local_class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(root, partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = local_class_choice

        if self.class_choice is not None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        local_label = self.label[item]
        # add 1 hot label
        one_hot_label = get_label_1_hot(local_label).squeeze()
        seg_label = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
            seg_label = seg_label[indices]
        return point_cloud, local_label, seg_label, one_hot_label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart(Dataset):
    # modified version for training encoder
    def __init__(self, root, num_points, partition='trainval', local_class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(root, partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = local_class_choice
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

        if self.class_choice is not None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        local_label = self.label[item]
        seg_label = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
            seg_label = seg_label[indices]
        _, morph1 = self.aug1(point_cloud)
        _, morph2 = self.aug2(point_cloud)
        return morph1, morph2, local_label, seg_label

    def __len__(self):
        return self.data.shape[0]


class ShapeNetPart_Scale(Dataset):
    # modified version for training encoder
    def __init__(self, root, num_points, scale_rate=0.01,
                 partition='trainval', local_class_choice=None):
        self.data, self.label, self.seg = load_data_partseg(root, partition, scale=scale_rate)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = local_class_choice
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

        if self.class_choice is not None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        local_label = self.label[item]
        seg_label = self.seg[item][:self.num_points]
        if self.partition == 'trainval':
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
            seg_label = seg_label[indices]
        _, morph1 = self.aug1(point_cloud)
        _, morph2 = self.aug2(point_cloud)
        return morph1, morph2, local_label, seg_label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    base_root = r'/Volumes/Haruki2022/PointCCA/data/part_seg/'


    def test_dataset(dataset, root=base_root):
        print('Test dataset %s ' % dataset.__name__)
        start = time.time()
        trainval = dataset(root=root, num_points=2048, partition='trainval')
        morph1, morph2, test_label, test_seg = trainval[0]
        print(morph1.shape)  # (2048, 3)
        print(test_label.shape)  # (1,)
        print(test_seg.shape)  # (2048, )
        print('Time: ', time.time() - start)  # Time:  1.96


    def test_dataloader(dataset, root=base_root):
        print('Test dataloader %s ' % dataset.__name__)
        train_dataset = dataset(
            root=root,
            partition='test',
            num_points=2048,
            scale_rate=0.05,
            local_class_choice=None)

        train_loader = DataLoader(train_dataset,
                                  num_workers=8,
                                  batch_size=2,
                                  shuffle=True,
                                  drop_last=True)
        print('Length of train loader is: ', train_loader.__len__())
        for m1, m2, label, seg in train_loader:
            start = time.time()
            print(m1.shape)  # torch.Size([2, 2048, 3])
            print('Time is: ', time.time() - start)  # 1.69
            break


    test_dataset(dataset=ShapeNetPart_Scale)
    test_dataloader(dataset=ShapeNetPart_Scale)

