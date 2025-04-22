import argparse
import logging
import os
import random
import shutil
import sys
import time

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

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler, datasetModelSegwithopencv)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume

parser = argparse.ArgumentParser()  # 管理命令行参数
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')  # 数据根路径
parser.add_argument('--exp', type=str,
                    default='ACDC/Mean_Teacher', help='experiment_name')    # 实验名称
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')  # 指定模型名称，默认为'unet'
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')    # 指定训练的最大迭代次数，默认为30000。
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')  # 指定每个GPU上的批处理大小，默认为24。
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')  # 用于确定是否使用确定性训练，默认为1，通常表示启用。
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')  # 指定分割网络的学习率，默认为0.01。
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')  # 指定网络输入的补丁大小，默认为[256, 256]。
parser.add_argument('--seed', type=int,  default=1337, help='random seed')  # 指定随机种子，默认为1337，这有助于实验的可重复性。
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')   # 用于指定网络的输出通道数量，默认为4，这通常是分类任务中类别数量。

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')  # 用于指定每个GPU上标记数据的批处理大小，默认为12。
parser.add_argument('--labeled_num', type=int, default=136,
                    help='labeled data')    # 用于指定使用的标记数据量，默认为136。
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')  # 指定指数移动平均的衰减率，默认为0.99
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')  # 指定一致性损失的类型，默认为"mse"，即均方误差
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')    # 指定一致性损失的权重，默认为0.1
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')   # 指定一致性损失权重逐渐增加到完全权重所需的迭代次数，默认为200.0
args = parser.parse_args()  # 解析命令行参数，并将它们存储在args变量中


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
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    """
    在训练过程中更新模型的指数移动平均（Exponential Moving Average，EMA）版本的权重
        model：当前正在训练的模型。（学生）
        ema_model：模型的指数移动平均版本，通常与model有相同的架构，但权重由EMA算法维护。（老师）
        alpha：EMA的平滑因子，决定了新旧权重的混合比例。通常，alpha接近1意味着更多依赖于旧的权重，接近0则更多依赖于新的权重。
        global_step：全局训练步骤计数，即模型已经进行了多少次梯度更新。
    """
    # 随着训练的进行，EMA权重将越来越依赖于旧的权重，这是为了在训练早期给予新权重更多的权重，而在训练后期更加平滑地融合权重变化。
    # 调整了alpha的值，使其不会超过1。随着global_step的增加，1 - 1 / (global_step + 1)的值逐渐接近1，因此alpha也会逐渐增大
    alpha = min(1 - 1 / (global_step + 1), alpha)

    # 遍历ema_model和model的所有参数。zip函数将两个模型的参数一一对应起来，使得可以同时访问原模型和EMA模型的参数
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # mul_方法将ema_param的值乘以alpha，这是保持EMA权重平滑的部分
        # add_方法将param的值乘以1 - alpha，然后加到经过乘法操作后的ema_param上。
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    """
    训练过程
        snapshot_path：保存模型快照的路径
    """
    base_lr = args.base_lr  # 学习率
    num_classes = args.num_classes  # 分类任务的类别数量
    batch_size = args.batch_size    # 训练批次大小
    max_iterations = args.max_iterations    # 最大训练迭代次数

    def create_model(ema=False):
        """创建模型实例，接受一个布尔参数ema，用于指示是否创建用于指数移动平均的模型"""
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)  # 模型类型由args.model决定
        # 创建ema模型
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()  # 学生模型
    ema_model = create_model(ema=True)  # 老师模型

    def worker_init_fn(worker_id):
        """工作线程初始化函数，用于设置每个线程的随机种子，确保数据增强的可重复性"""
        random.seed(args.seed + worker_id)

    # 创建 训练 数据集实例：数据集路径为args.root_path；
    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    # 创建 验证 数据集实例
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)    # 总样本数
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)    # 有标签的样本数
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))

    labeled_idxs = list(range(0, labeled_slice))    # 用 含标签样本数 构成一个列表
    unlabeled_idxs = list(range(labeled_slice, total_slices))   # 用 无标签样本数 构成一个列表

    # 用于半监督学习的特殊采样器，它可以同时从标记和未标记数据中采样
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    # 创建训练数据加载器，使用上述批采样器，并设置数据加载线程数为4，启用pin memory优化
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    model.train()  # 将模型设置为训练模式

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)   # 创建验证数据加载器，通常用于评估模型性能，使用批大小为1

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)    # 创建随机梯度下降（SGD）优化器，用于模型参数的更新
    ce_loss = CrossEntropyLoss()    # 创建交叉熵损失函数实例
    dice_loss = losses.DiceLoss(num_classes)    # 创建Dice损失函数实例，用于评估模型的分割性能

    writer = SummaryWriter(snapshot_path + '/log')   # 创建TensorBoard日志记录器，用于可视化训练过程中的指标
    logging.info("{} iterations per epoch".format(len(trainloader)))  # 输出每个epoch的迭代次数

    iter_num = 0    # 初始化迭代计数器
    max_epoch = max_iterations // len(trainloader) + 1  # 计算最大训练轮数，基于最大迭代次数和每轮的迭代次数
    best_performance = 0.0  # 初始化最佳性能指标
    iterator = tqdm(range(max_epoch), ncols=70)  # 创建进度条，用于显示训练过程中的进度，ncols=70设置了进度条的宽度

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # 获取数据
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()  # 将数据移到GPU
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]     # 无标签的数据batch

            # 加噪声
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            # 学生模型正常向前传播并得到输出概率分布
            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            # 老师模型不进行梯度计算（不进行反向传播），而是正常向前传播并输出结果及其概率分布
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)


            loss_ce = ce_loss(outputs[:args.labeled_bs], label_batch[:][:args.labeled_bs].long())
            # 有标签数据的交叉熵损失

            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            # 有标签数据的dice损失

            supervised_loss = 0.5 * (loss_dice + loss_ce)  # 加权获得总损失

            # 根据迭代次数来计算一致性损失权重
            consistency_weight = get_current_consistency_weight(iter_num//150)
            if iter_num < 1000:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)
            # 如果迭代次数小于 1000，则一致性损失为 0，否则计算未标记数据的一致性损失

            loss = supervised_loss + consistency_weight * consistency_loss
            # 将监督损失和加权的一致性损失相加，得到最终的总损失。

            optimizer.zero_grad()   # 清除优化器中所有参数的梯度。
            loss.backward()  # 对总损失进行反向传播，计算梯度。
            optimizer.step()   # 根据计算的梯度更新模型参数。
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)    # 更新 EMA 模型的参数，以平滑模型的权重。
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9  # 根据当前的迭代次数调整学习率。

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            # 更新优化器中的学习率。

            iter_num = iter_num + 1  # 更新迭代次数。

            writer.add_scalar('info/lr', lr_, iter_num)             # 记录学习率
            writer.add_scalar('info/total_loss', loss, iter_num)    # 当前的总损失
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)    # 交叉熵损失
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)    # dice损失
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)  # 一致性损失
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)  # 当前迭代次数和损失信息

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))  # 使用 logging 记录当前迭代次数和损失信息。

            if iter_num % 20 == 0:
                # 每 20 次迭代执行以下：
                image = volume_batch[1, 0:1, :, :]  # 从批次数据中提取一个示例图像：第二个batch第一个通道的图像
                writer.add_image('train/Image', image, iter_num)    # 将示例图像添加'train/Image'下。
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)   # 将模型的输出转换为预测标签
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)  # 将预测的标签图像添加到'train/Prediction'
                labs = label_batch[1, ...].unsqueeze(0) * 50    # 将真实标签转换为图像格式
                writer.add_image('train/GroundTruth', labs, iter_num)   # 将真实标签图像添加到'train/GroundTruth'下

            if iter_num > 0 and iter_num % 200 == 0:
                # 当迭代次数大于 0 且每 200 次迭代执行以下
                model.eval()    # 评估模式
                metric_list = 0.0   # 存储验证集指标
                for i_batch, sampled_batch in enumerate(valloader):
                    # 遍历验证数据集中的每一个批次
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)  # 计算当前批次的指标
                    metric_list += np.array(metric_i)    # 累加指标
                metric_list = metric_list / len(db_val)  # 计算平均指标

                for class_i in range(num_classes-1):
                    # 对于每个类别，将 Dice 和 HD95 指标添加到 TensorBoard
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]   # 计算平均 Dice 指标
                mean_hd95 = np.mean(metric_list, axis=0)[1]  # 计算平均 HD95 指标
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    # 如果当前性能超过最佳性能，保存模型
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    # 设置cuDNN
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False  # 提高运行效率，但每次运行的结果可能会有所不同
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True  # 确保每次运行的结果都是可复现的，但可能会牺牲一些性能

    # 设置随机数生成器的种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)  # 模型快照路径
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
