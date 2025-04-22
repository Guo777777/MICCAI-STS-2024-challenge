import argparse
import logging
import os
import random
import shutil
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import imageio
import datetime

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler, datasetModelSegwithopencv)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

def patients_to_slices(dataset, patiens_num):
    """根据给定的数据集名称和患者数量，返回该数据集中对应的切片数量"""
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    """在训练过程中根据当前的epoch（训练轮次）动态计算一致性损失的权重"""  # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # args.consistency：一致性损失的权重，默认为0.1
    # ramps.sigmoid_rampup：用于实现Sigmoid型的权重上升曲线
    # args.consistency_rampup：一致性损失权重逐渐增加到完全权重所需的迭代次数，默认为200.0
    consistency = 0.1
    consistency_rampup = 200
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)

    # 遍历ema_model和model的所有参数。zip函数将两个模型的参数一一对应起来，使得可以同时访问原模型和EMA模型的参数
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # mul_方法将ema_param的值乘以alpha，这是保持EMA权重平滑的部分
        # add_方法将param的值乘以1 - alpha，然后加到经过乘法操作后的ema_param上。
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(snapshot_path, base_lr, num_classes, batch_size, max_iterations,patch_size,
          labeled_num, labeled_bs, net_type, seed=1337,
          trainimages=None, trainlabels=None, valimages=None, vallabels=None,
          modelpath=None):


    def create_model(ema=False):
        """创建模型实例，接受一个布尔参数ema，用于指示是否创建用于指数移动平均的模型"""
        # Network definition
        model = net_factory(net_type=net_type, in_chns=1,
                            class_num=num_classes)
        if ema:  # 创建ema模型
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()  # 学生模型
    ema_model = create_model(ema=True)  # 老师模型

    device = torch.device("cuda:0")
    if modelpath:
        model.load_state_dict(torch.load(modelpath, map_location=device))
        ema_model.load_state_dict(torch.load(modelpath, map_location=device))

    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)

    db_train = datasetModelSegwithopencv(trainimages, trainlabels, split="train",
                                         transform=transforms.Compose([RandomGenerator(patch_size)]))
    db_val = datasetModelSegwithopencv(valimages, vallabels, split="val")
    total_slices = len(db_train)
    labeled_slice = labeled_num
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)   # 创建验证数据加载器，通常用于评估模型性能，使用批大小为1

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)    # 创建随机梯度下降（SGD）优化器，用于模型参数的更新
    ce_loss = CrossEntropyLoss()    # 创建交叉熵损失函数实例
    dice_loss = losses.DiceLoss(num_classes)    # 创建Dice损失函数实例，用于评估模型的分割性能

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0    # 初始化迭代计数器
    max_epoch = max_iterations // len(trainloader) + 1  # 计算最大训练轮数，基于最大迭代次数和每轮的迭代次数
    best_performance = 0.0  # 初始化最佳性能指标
    iterator = tqdm(range(max_epoch), ncols=70)  # 进度条

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # 获取数据
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()  # 将数据移到GPU

            unlabeled_volume_batch = volume_batch[labeled_bs:]     # 无标签的数据batch
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.2, -0.4, 0.4)
            ema_inputs = unlabeled_volume_batch + noise  # 经过强化的volume_batch[labeled_bs:]
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)  # 老师输出
                ema_output_soft = torch.softmax(ema_output, dim=1)

            outputs = model(volume_batch)   # 学生输出
            outputs_labeledbs_soft = torch.softmax(outputs[:labeled_bs], dim=1)  # 学生有标签输出的概率分布
            loss_ce = ce_loss(outputs[:labeled_bs], label_batch[:][:labeled_bs].float())  # 有标签数据的交叉熵损失
            loss_dice = dice_loss(outputs_labeledbs_soft, label_batch[:labeled_bs].float())  # 有标签数据的dice损失（使用概率）
            supervised_loss = 0.5 * (loss_dice + loss_ce)  # 加权获得总损失

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            # if iter_num < 1000:
            #     consistency_loss = 0.0
            # else:
            # outputs_check = torch.all((outputs_soft[labeled_bs:]) == 0)
            # emaoutputs_check = torch.all(ema_output_soft == 0)
            # consistency_dist = losses.softmax_mse_loss(outputs_soft[labeled_bs:], ema_output_soft)
            # else:
            outputs_unlabedbs_soft = torch.softmax(outputs[labeled_bs:], dim=1)  # 学生无标签输出的概率分布
            consistency_dist = torch.mean((outputs_unlabedbs_soft - ema_output_soft) ** 2)
            consistency_loss = consistency_dist * consistency_weight
            loss = supervised_loss + consistency_loss

            optimizer.zero_grad()   #
            loss.backward()
            optimizer.step()
            ema_decay = 0.99
            update_ema_variables(model, ema_model, ema_decay, iter_num)    # 更新 EMA 模型的参数，以平滑模型的权重。
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  # 根据当前的迭代次数调整学习率。

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/lr', lr_, iter_num)             # 记录学习率
            writer.add_scalar('info/total_loss', loss, iter_num)    # 当前的总损失
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)    # 交叉熵损失
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)    # dice损失
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)  # 一致性损失
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)  # 当前迭代次数和损失信息

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, supervised_loss: %f, consistency_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), supervised_loss, consistency_loss))

            # if epoch_num >= 250 and iter_num % len(trainloader) == 0:
            if iter_num % len(trainloader) == 0:
                dice_total = 0
                hd95_total = 0
                save_dir = os.path.join(snapshot_path, 'val_pic')
                os.makedirs(save_dir, exist_ok=True)
                for i_batch, sampled_batch in enumerate(valloader):
                    dice, hd95, pred = test_single_volume(sampled_batch["image"], sampled_batch["label"],
                                                          model, num_classes=num_classes, patch_size=[1024, 1024])
                    dice_total += dice
                    hd95_total += hd95

                    # for i in range(pred.shape[0]-1):
                    #     pred[i, :, :] *= (i + 1)

                    height, width = pred.shape[1], pred.shape[2]
                    Masknp = np.zeros((num_classes, height, width), np.uint8)
                    for label_index in range(1, num_classes):
                        masknp = np.zeros((height, width), dtype=np.uint8)
                        masknp[pred[label_index] == 1] = label_index  # 为每个标签设置不同的灰度值
                        Masknp[label_index] = masknp

                    foreground_mask = np.max(Masknp[1:num_classes, :, :], axis=0)
                    Masknp[0, :, :] = 1 - foreground_mask
                    pred_background = Masknp[0, :, :]
                    pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
                    Masknp[0, :, :] = pred_background
                    prediction = np.sum(Masknp, axis=0)
                    prediction_uint8 = (prediction * 255).astype(np.uint8)

                    save_path = os.path.join(save_dir, 'iter_{}_dice_{}.jpg'.format(iter_num, round(dice, 4)))
                    imageio.imwrite(save_path, prediction_uint8)

                dice_avg = dice_total / len(valloader)
                hd95_avg = hd95_total / len(valloader)

                performance = dice_avg
                mean_hd95 = hd95_avg
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    # 如果当前性能超过最佳性能，保存模型
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'epoch_{}_dice_{}.pth'.format(epoch_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(net_type))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break

            iter_num = iter_num + 1  # 更新迭代次数。

        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    # 设置cuDNN
    torch.cuda.empty_cache()
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False  # 提高运行效率，但每次运行的结果可能会有所不同
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True  # 确保每次运行的结果都是可复现的，但可能会牺牲一些性能

    # 获取数据集
    data_dir = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/data/NewDataset/TrainDataset.csv'
    csv_data = pd.read_csv(data_dir)
    trainimages = csv_data.iloc[:, 0].values
    trainlabels = csv_data.iloc[:, 1].values
    # data_dir2 = 'dataprocess/data/testseg.csv'
    data_dir2 = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/data/NewDataset/ValDataset.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values


    # 设置随机数生成器的种子
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    snapshot_dir = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/model_log'
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = os.path.join(snapshot_dir, current_time)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))

    model_name = 'unet'  # 使用网络
    base_lr = 0.01  # 学习率
    num_classes = 53  # 分类任务的类别数量
    batch_size = 8   # 训练批次大小
    max_iterations = 56000    # 最大训练迭代次数
    root_path = None
    patch_size = [1024, 1024]
    labeled_num = 46  # 有标签样本数量 138/23=6
    labeled_bs = 4  # 处理有标签样本batch

    model_path = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/model_log/20240828_112113/unet_best_model.pth'

    print(data_dir, data_dir2, model_path)

    train(snapshot_path=snapshot_path,
          base_lr=base_lr,
          num_classes=num_classes,
          batch_size=batch_size,
          max_iterations=max_iterations,
          patch_size=patch_size,
          labeled_num=labeled_num,
          labeled_bs=labeled_bs,
          net_type=model_name,
          trainimages=trainimages,
          trainlabels=trainlabels,
          valimages=valimages,
          vallabels=vallabels,
          modelpath=model_path
          )
