import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from data.augmentation import PointWOLF


def load_data_semseg(base_dir, partition, test_area):
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    if partition == 'train':
        data_dir = os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batch_list, label_batch_list = [], []
    for f in all_files:
        file = h5py.File(os.path.join(os.path.dirname(data_dir), f), 'r+')
        h5_data = file["data"][:]
        label = file["label"][:]
        data_batch_list.append(h5_data)
        label_batch_list.append(label)
    data_batches = np.concatenate(data_batch_list, 0)
    seg_batches = np.concatenate(label_batch_list, 0)
    test_area_name = "Area_" + test_area
    train_idx, test_idx = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idx.append(i)
        else:
            train_idx.append(i)
    if partition == 'train':
        all_data = data_batches[train_idx, ...]
        all_seg = seg_batches[train_idx, ...]
    else:
        all_data = data_batches[test_idx, ...]
        all_seg = seg_batches[test_idx, ...]
    return all_data, all_seg


class S3DIS(Dataset):
    """
    output dim: [4096, 9]
    """
    def __init__(self, root, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(root, partition, test_area)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        seg_label = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
            seg_label = seg_label[indices]
        seg_label = torch.LongTensor(seg_label)
        return point_cloud, seg_label

    def __len__(self):
        return self.data.shape[0]


class S3DIS_AUG(Dataset):
    """
    output dim: [4096, 6], [4096, 6]
    """
    def __init__(self, root, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(root, partition, test_area)
        self.num_points = num_points
        self.partition = partition
        self.aug1 = PointWOLF(0.4)
        self.aug2 = PointWOLF(0.8)

    def __getitem__(self, item):
        point_cloud = self.data[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(point_cloud.shape[0]))
            np.random.shuffle(indices)
            point_cloud = point_cloud[indices]
        # N, 9 ---> N, 3
        pc_coord = point_cloud[:, :3]
        pc_feat = point_cloud[:, 3:6]
        _, morph1 = self.aug1(pc_coord)
        _, morph2 = self.aug2(pc_coord)
        morph1 = np.concatenate((morph1, pc_feat), axis=-1)
        morph2 = np.concatenate((morph2, pc_feat), axis=-1)
        return morph1, morph2

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    import time
    from torch.utils.data import DataLoader

    base_root = '/home/haruki/下载/PointCCA/data/sem_seg'


    def test_dataset(dataset_class, root=base_root):
        print('- Dataset Test for %s...' % str(dataset_class))
        start_time = time.time()
        train = dataset_class(root, 4096, 'train')
        data, seg = train[0]
        print(data.shape)  # (4096, 9)
        print(seg.shape)  # torch.Size([4096])
        print('Read Data Time: ', time.time() - start_time)  # 35.21


    def test_dataloader(dataset_class, root=base_root):
        print('- Dataloader Test for %s...' % str(dataset_class))
        train_loader = DataLoader(
            dataset_class(root=root,
                          partition='train',
                          num_points=4096,
                          test_area='5'),
            num_workers=8,
            batch_size=2,
            shuffle=True,
            drop_last=True
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        start_time = time.time()
        for m1, m2 in train_loader:
            m1, m2 = m1.to(device), m2.to(device)
            # [2, 4096, 6] --- B, N, f
            print(m1.shape, m2.shape)
            # Read 1 batch time: 3.19
            print('Read 1 batch time: ', time.time() - start_time)
            break


    # test1: original dataset test
    test_dataset(dataset_class=S3DIS, root=base_root)
    # test2: dataloader test
    test_dataset(dataset_class=S3DIS_AUG, root=base_root)
    # test3: augmented dataset test:
    test_dataloader(dataset_class=S3DIS, root=base_root)
    # test4: augmented dataloader:
    test_dataloader(dataset_class=S3DIS_AUG, root=base_root)
