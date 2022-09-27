from partseg_hf5 import ShapeNetPart_Original
from partseg_test.partseg_classifier import PartSegClassifier, PartSegClassifier_2
from utils.train_partseg_with_IoU import calculate_shape_IoU, calculate_category_IoU
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
from tqdm import tqdm
import math

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model, use pretrained weights
model_path = r'/home/haruki/下载/SimAttention/scripts/psg_weights/model_seg_partseg_3_98.pth'
# model = PartSegClassifier(model_path).to(device)
model = PartSegClassifier_2(model_path).to(device)


# parameters
lr = 1e-3
lrf = 1e-3
max_epoch = 400
batch_size = 12
criterion = torch.nn.CrossEntropyLoss()

# load data
root = r'/home/haruki/下载/shapenet/shapenet_part_seg_hdf5_data'
train_data_set = ShapeNetPart_Original(
    root=root,
    num_points=2048,
    partition='trainval',
    class_choice=None
)
train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
train_loader = tqdm(train_loader)
test_data_set = ShapeNetPart_Original(
    root=root,
    num_points=2048,
    partition='test',
    class_choice=None
)
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, shuffle=True)
test_loader = test_loader

# log parameters
log_file = r'/home/haruki/下载/SimAttention/partseg_test/runs/local_test_04'
if not os.path.exists(log_file):
    os.makedirs(log_file)
tb_writer = SummaryWriter(log_dir=log_file)
tags = ["learning_rate",
        "instance_train_mIoU",
        "instance_test_mIoU",
        "category_train_mIoU",
        "category_test_mIoU",
        "train_acc",
        "test_acc",
        "train_avg_per_class_acc",
        "test_avg_per_class_acc",
        "train_loss",
        "test_loss"]


@torch.no_grad()
def evaluate(test_model, loader):
    test_loss = 0.0
    count = 0.0
    test_model.eval()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []

    for feature, label, seg, label_1_hot in loader:
        feature, seg, label_1_hot = feature.to(device), seg.to(device), label_1_hot.to(device)
        pred = test_model(feature.float(), label_1_hot)
        loss = criterion(pred, seg.long())
        pred_choice = pred.data.max(1)[1]
        test_batch_size = feature.shape[0]
        count += test_batch_size
        test_loss += loss.item() * test_batch_size
        seg_np = seg.cpu().numpy()
        pred_np = pred_choice.detach().cpu().numpy()

        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ins_IoUs = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, None)
    test_category_ious_dict = calculate_category_IoU(test_ins_IoUs, test_label_seg)
    return test_acc, test_loss * 1.0 / count, np.mean(test_ins_IoUs), avg_per_class_acc, test_category_ious_dict


def train_1_epoch(train_model, loader, optimizer, epoch):
    train_model.train()
    # for epoch in range(0, max_epoch):
    print('\n')
    print('***** {} epoch now is training... ******'.format(epoch))
    train_loss = 0.0
    count = 0.0
    train_true_cls = []
    train_pred_cls = []
    train_true_seg = []
    train_pred_seg = []
    train_label_seg = []

    for feature, label, seg, label_1_hot in loader:
        feature, seg, label_1_hot = feature.to(device), seg.to(device), label_1_hot.to(device)
        optimizer.zero_grad()
        # print('label 1 hot shape: ', label_1_hot.shape)
        pred = train_model(feature.float(), label_1_hot)  # (B, seg_num_all, N)
        loss = criterion(pred, seg.long())
        loss.backward()
        optimizer.step()

        # seg_pred = pred.permute(0, 2, 1).contiguous()
        # pred_choice = seg_pred.max(dim=2)[1]
        pred_choice = pred.data.max(1)[1]
        train_batch_size = feature.shape[0]
        count += train_batch_size
        train_loss += loss.item() * train_batch_size
        # train_loss = train_loss * 1.0 / count
        seg_np = seg.cpu().numpy()
        pred_np = pred_choice.detach().cpu().numpy()
        train_true_cls.append(seg_np.reshape(-1))
        train_pred_cls.append(pred_np.reshape(-1))

        train_true_seg.append(seg_np)
        train_pred_seg.append(pred_np)
        train_label_seg.append(label.reshape(-1))

    train_true_cls = np.concatenate(train_true_cls)
    train_pred_cls = np.concatenate(train_pred_cls)
    train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
    train_true_seg = np.concatenate(train_true_seg, axis=0)
    train_pred_seg = np.concatenate(train_pred_seg, axis=0)
    train_label_seg = np.concatenate(train_label_seg)
    # print('train_label_seg shape: ', len(train_label_seg))  # 14007
    train_ins_IoUs = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, None)
    # print('train_IoUs shape: ', len(train_IoUs), train_IoUs[0])  # 14007
    train_category_ious_dict = calculate_category_IoU(train_ins_IoUs, train_label_seg)
    # print('dict: ', train_category_ious_dict)
    return train_acc, train_loss * 1.0 / count, np.mean(train_ins_IoUs), avg_per_class_acc, train_category_ious_dict


def train_test_record(epochs):
    # sgd opt
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    # adamw opt
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    lf = lambda x: ((1 + math.cos(x * math.pi / max_epoch)) / 2) * (1 - lrf) + lrf
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # train and test category shape mIoUs dict
    class_choices = [
        'airplane', 'bag', 'cap', 'car',
        'chair', 'earphone', 'guitar', 'knife',
        'lamp', 'laptop', 'motorbike', 'mug',
        'pistol', 'rocket', 'skateboard', 'table'
    ]
    train_best_category_dict = {}
    test_best_category_dict = {}
    for i in range(len(class_choices)):
        train_best_category_dict[class_choices[i]] = 0.0
        test_best_category_dict[class_choices[i]] = 0.0

    try:
        checkpoint = torch.load('/home/haruki/下载/SimAttention/psg_data/best_ins-IoU_psg_98_sgd_update_400_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_test_ins_ious = checkpoint['test_ins_ious']
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        best_test_ins_ious = 0.0

    for epoch in range(epochs):
        train_acc, train_loss, ins_train_iou, train_avg_acc, train_category_dict = train_1_epoch(
            train_model=model,
            optimizer=optimizer,
            loader=train_loader,
            epoch=epoch)

        scheduler.step()

        test_acc, test_loss, ins_test_iou, test_avg_acc, test_category_dict = evaluate(test_model=model,
                                                                                       loader=test_loader)
        print('- {} epoch test instance mIoU is: '.format(epoch), ins_test_iou)

        # category mIoU, above show all single category mIoU
        train_cate_mious = sum(train_category_dict.values()) / 16
        test_cate_mious = sum(test_category_dict.values()) / 16
        print('- {} epoch test category mIoU is: '.format(epoch), test_cate_mious)

        # get the best value for each category
        for i in range(len(class_choices)):
            if train_best_category_dict[class_choices[i]] < train_category_dict[class_choices[i]]:
                train_best_category_dict[class_choices[i]] = train_category_dict[class_choices[i]]
            if test_best_category_dict[class_choices[i]] < test_category_dict[class_choices[i]]:
                test_best_category_dict[class_choices[i]] = test_category_dict[class_choices[i]]

        tb_writer.add_scalar(tags[0], optimizer.param_groups[0]["lr"], epoch)
        # add instance_mIoU + single_category_mIoU + category_mIoU:  1 and 3 need curve on tensorboard, 2 just record
        tb_writer.add_scalar(tags[1], ins_train_iou, epoch)
        tb_writer.add_scalar(tags[2], ins_test_iou, epoch)
        tb_writer.add_scalar(tags[3], train_cate_mious, epoch)
        tb_writer.add_scalar(tags[4], test_cate_mious, epoch)
        tb_writer.add_scalar(tags[5], train_acc, epoch)
        tb_writer.add_scalar(tags[6], test_acc, epoch)
        tb_writer.add_scalar(tags[7], train_avg_acc, epoch)
        tb_writer.add_scalar(tags[8], test_avg_acc, epoch)
        tb_writer.add_scalar(tags[9], train_loss, epoch)
        tb_writer.add_scalar(tags[10], test_loss, epoch)

        print('Train Best Category mIoUs Dict: ')
        print(train_best_category_dict)
        print('Test Best Category mIoUs Dict: ')
        print(test_best_category_dict)

        if best_test_ins_ious < ins_test_iou:
            best_test_ins_ious = ins_test_iou
            print('{} epoch save model...'.format(epoch))
            save_path = '/home/haruki/下载/SimAttention/psg_data/best_ins-IoU_psg_98_sgd_update_400_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'test_ins_ious': best_test_ins_ious,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)
        print('best test instance IoU now is: ', best_test_ins_ious)


if __name__ == "__main__":
    train_test_record(max_epoch)
