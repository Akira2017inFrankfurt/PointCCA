import os
import math
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataloader import LatentRepresentationDataSet
from network.shape_classifier_2 import ShapeClassifier_2


# train and test data loader
def get_loader(root):
    train_root = os.path.join(root, 'train')
    test_root = os.path.join(root, 'test')
    train_lrds = LatentRepresentationDataSet(train_root)
    test_lrds = LatentRepresentationDataSet(test_root)
    train_loader = torch.utils.data.DataLoader(train_lrds, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_lrds, batch_size=1, shuffle=True)
    # delete the tqdm function
    return train_loader, test_loader


# log file
def log_func(log_file):
    if not os.path.exists(log_file):
        os.makedirs(log_file)
        print("Make Log File! ", log_file)
    else:
        print("Log File Already Exists")
    tensorboard_writer = SummaryWriter(log_dir=log_file)
    return tensorboard_writer


# evaluate classifier on test dataset
@torch.no_grad()
def evaluate_model(model, loader, device):
    test_mean_correct = []
    test_loss = 0.0
    for f, l in loader:
        f, l = f.to(device), l.to(device)
        # use half features
        f = f.reshape(-1, 1024)
        l = l.reshape(l.shape[-1] * l.shape[0])  # torch.Size([8])
        cls = model.eval()
        pred = cls(f.float())
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(pred, l.long())
        test_loss += loss.item()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(l.long().data).cpu().sum()
        test_mean_correct.append(correct.item() / float(f.size()[0]))
    test_instance_acc = np.mean(test_mean_correct)
    return test_instance_acc, test_loss / len(loader)


# major function
def train_and_test(device, tb_writer, trainDataLoader, testDataLoader, args):
    learning_rate = args.lr
    learning_rate_final = args.lrf
    max_epochs = args.epoch
    opt_choice = args.opt_choice
    lr_strategy = args.lr_strategy
    classifier = ShapeClassifier_2().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_cls_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        best_test_acc = checkpoint['test_acc']
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        best_test_acc = 0.0
        start_epoch = 0

    if opt_choice == 0:
        optimizer = torch.optim.SGD(classifier.parameters(),
                                    lr=learning_rate,
                                    momentum=0.9,
                                    weight_decay=0.005)
    elif opt_choice == 1:
        # new optimizer AdamW
        optimizer = torch.optim.AdamW(classifier.parameters(),
                                      lr=learning_rate,
                                      betas=(0.9, 0.999),
                                      weight_decay=0.01)

    if lr_strategy == 0:
        lf = lambda x: ((1 + math.cos(x * math.pi / max_epochs)) / 2) * (1 - learning_rate_final) + learning_rate_final
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif lr_strategy == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
    tags = ["train_acc", "learning_rate", "train_loss", "test_acc", "test_loss"]

    # record the best information
    pair_train_acc = 0.0
    best_test_epoch = 0

    for epoch in range(0, max_epochs):
        # train mean correct
        mean_correct = []
        train_loss = 0.0
        for f, l in trainDataLoader:
            f, l = f.to(device), l.to(device)
            # what if only use 512 features
            f = f.reshape(-1, 1024)  # torch.Size([8, 1024])
            l = l.reshape(l.shape[-1] * l.shape[0])  # torch.Size([8])

            optimizer.zero_grad()
            pred = classifier(f.float())  # torch.Size([8, 40])
            loss = criterion(pred, l.long())
            train_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(l.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(f.size()[0]))
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)

        tb_writer.add_scalar(tags[0], train_instance_acc, epoch)
        tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[2], train_loss / len(trainDataLoader), epoch)

        # evaluation part
        test_acc, test_loss = evaluate_model(classifier.eval(), testDataLoader, device)
        tb_writer.add_scalar(tags[3], test_acc, epoch)
        tb_writer.add_scalar(tags[4], test_loss, epoch)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            pair_train_acc = train_instance_acc
            best_test_epoch = epoch
            print('Save model...')
            save_path = 'best_cls_model.pth'
            print('Saving at %s' % save_path)
            state = {
                'epoch': best_test_epoch,
                'test_acc': best_test_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_path)
    print("Best test_acc is {}, train_acc is {}, epoch is {}".format(best_test_acc, pair_train_acc, best_test_epoch))
    print("***********\n")


def run(args):
    print("Chosen model is: ", args.data_file[38:])
    log_file = os.path.join(args.data_file, args.run_file_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data_loader, test_data_loader = get_loader(args.data_file)

    train_and_test(device,
                   log_func(log_file),
                   train_data_loader,
                   test_data_loader,
                   args)

    # direct see the results without tensorboard command
    # command_str = 'tensorboard --logdir={}'.format(log_file)
    # os.system(command_str)


if __name__ == '__main__':
    # basic setup
    optimizer_list = [0, 1]
    opt_name_list = ['sgd', 'adamw']
    opt_choice = 0
    # dataset basic name and index
    dataset_name = r'model_knn_2048_proj_1_4_1-v2-'
    name_list = [110]
    train_epochs = [100, 200, 300, 400]
    for train_epoch in train_epochs:
        log_file_name = 'run_' + opt_name_list[opt_choice] + '_' + str(train_epoch)
        print('Start training...')
        for name_index in name_list:
            dataset_name += str(name_index)
            parser = argparse.ArgumentParser()
            parser.add_argument('--data_file', type=str,
                                default=os.path.join('/home/haruki/下载/SimAttention/cls_data/', dataset_name))
            parser.add_argument('--lr', type=float, default=0.0001)
            parser.add_argument('--lrf', type=float, default=0.01)
            parser.add_argument('--epoch', type=int, default=train_epoch)
            parser.add_argument('--opt_choice', type=int, default=opt_choice)
            parser.add_argument('--lr_strategy', type=int, default=0)
            parser.add_argument('--run_file_name', type=str, default=log_file_name)
            opt = parser.parse_args()

            run(opt)

            if name_index < 10:
                dataset_name = dataset_name[:-1]
            elif name_index < 100:
                dataset_name = dataset_name[:-2]
            else:
                dataset_name = dataset_name[:-3]

# choose best train epoch
# if __name__ == '__main__':
#     dataset_name = 'model_knn_2048_proj_1_4_1-v2-'
#     # name_list = [10, 15, 20, 25, 30, 40]
#     name_list = [110]
#     for name_index in name_list:
#         dataset_name += str(name_index)
#         epochs = [70]
#         for epoch in epochs:
#             print("Run {} epochs".format(epoch))
#             file_name = 'run_adamw_cos_30_' + str(epoch)
#             parser = argparse.ArgumentParser()
#             parser.add_argument('--data_file', type=str,
#                                 default=os.path.join('/home/haruki/下载/SimAttention/cls_data/', dataset_name))
#             parser.add_argument('--lr', type=float, default=0.0001)
#             parser.add_argument('--lrf', type=float, default=0.01)
#             parser.add_argument('--epoch', type=int, default=epoch)
#             parser.add_argument('--opt_choice', type=int, default=1)
#             parser.add_argument('--lr_strategy', type=int, default=1)
#             parser.add_argument('--run_file_name', type=str, default=file_name)
#             opt = parser.parse_args()
#
#             run(opt)
#
#         if name_index < 10:
#             dataset_name = dataset_name[:-1]
#         elif name_index < 100:
#             dataset_name = dataset_name[:-2]
#         else:
#             dataset_name = dataset_name[:-3]
