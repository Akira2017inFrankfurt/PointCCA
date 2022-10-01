import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from data.augmentation import PointWOLF


def random_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]
    return pc


def translate_point_cloud(point_cloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_point_cloud = np.add(np.multiply(point_cloud, xyz1), xyz2).astype('float32')
    return translated_point_cloud


class Modelnet40(Dataset):
    def __init__(self, root, split='train', cache_size=15000):
        super().__init__()
        self.root = root
        self.cat_file = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.cat_file)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        assert (split == 'train' or split == 'test')

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i],
                           os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                          in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.data_path)))
        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.data_path)

    def _get_item(self, index):
        if index in self.cache:
            morph1, morph2 = self.cache[index]
        else:
            fn = self.data_path[index]
            point_set = np.genfromtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = point_set[:, 0:3]

            point_set = random_dropout(point_set)
            point_set = translate_point_cloud(point_set)

            _, morph1 = self.aug1(point_set)
            _, morph2 = self.aug2(point_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (morph1, morph2)

        return morph1, morph2

    def __getitem__(self, index):
        return self._get_item(index)


class Modelnet40_Scale(Dataset):
    def __init__(self, root, split='train', scale_rate=0.01, cache_size=15000):
        super().__init__()
        self.root = root
        self.cat_file = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.cat_file)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

        shape_ids = {'train': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))],
                     'test': [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]}

        assert (split == 'train' or split == 'test')

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i],
                           os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                          in range(len(shape_ids[split]))]
        if split == 'train':
            self.scale_data_path = list(self.data_path)
            random.shuffle(self.scale_data_path)
            self.scale_data_path = self.scale_data_path[: int(scale_rate * len(self.data_path))+1]
            print('With {} scale_rate, the size of train data is {}'.format(scale_rate, len(self.scale_data_path)))
        else:
            self.scale_data_path = self.data_path
            print('The size of test data is %d' % ( len(self.scale_data_path)))
            scale_rate = 1
        self.cache_size = cache_size * scale_rate
        self.cache = {}

    def __len__(self):
        return len(self.scale_data_path)

    def _get_item(self, index):
        if index in self.cache:
            morph1, morph2 = self.cache[index]
        else:
            fn = self.scale_data_path[index]
            point_set = np.genfromtxt(fn[1], delimiter=',').astype(np.float32)
            point_set = point_set[:, 0:3]

            point_set = random_dropout(point_set)
            point_set = translate_point_cloud(point_set)

            _, morph1 = self.aug1(point_set)
            _, morph2 = self.aug2(point_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (morph1, morph2)

        return morph1, morph2

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == "__main__":
    base_root = r'/Volumes/Haruki2022/PointCCA/data/cls/data'


    def test_data(dataset):
        aug_dataset = dataset(root=base_root, split='train', scale_rate=0.05)
        trainDataLoader = torch.utils.data.DataLoader(aug_dataset, batch_size=4,
                                                      shuffle=True, num_workers=4,
                                                      pin_memory=True)
        count = 0
        for aug1, aug2 in trainDataLoader:
            count += 1
            print(count)
        print('test done')

    # test_data()  # torch.Size([4, 10000, 3]) ---> B, N, f
    # test_data(Modelnet40)
    test_data(Modelnet40_Scale)
