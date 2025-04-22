from scipy.ndimage import zoom
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from medpy import metric



class datasetModel_2d(Dataset):
    def __init__(self, images, labels, split="train", transform=None, ops_weak=None, ops_strong=None):
        super(datasetModel_2d).__init__()

        self.images = images  # 图像数据集
        self.labels = labels  # 标签数据集
        self.targetsize = (1024, 1024)    # 目标尺寸
        self.num_class = 53   # 分类数

        self.sample_list = []       # 样本列表
        self.split = split          # 数据集种类
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(ops_strong), \
            "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 获取样本
        imagesample = self.images[index]
        labelsample = self.labels[index]

        # 图像
        image = cv2.imread(imagesample, 0)
        origin_size = image.shape
        image = cv2.resize(image, (self.targetsize[0], self.targetsize[1]))
        image = (image - image.mean()) / image.std()
        images_tensor = torch.as_tensor(image).float()  # 转换为 PyTorch 张量
        # 标签
        mask = self.load_label_from_json(labelsample, self.targetsize)
        if mask is not None:
            label_tensor = torch.as_tensor(mask).float()
        else:
            label_tensor = None

        sample = {"image": images_tensor, "label": label_tensor}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):    # 应用数据增强
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        return sample

    def load_label_from_json(self, json_path, targetsize):
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        height, width = label_data['imageHeight'], label_data['imageWidth']
        Masknp = np.zeros((height, width), dtype=np.uint16)

        for shape in label_data['shapes']:
            label = shape['label']
            label_index = int(label)

            ten_digit = (int(label) // 10) % 10  # 牙齿标签的十位数
            if ten_digit <= 4:
                label_index = label_index - (11 + (ten_digit - 1) * 2)
            else:
                label_index = label_index - (19 + (ten_digit - 5) * 5)

            points = np.array(shape['points'], dtype=np.int32)
            points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly

            cv2.fillPoly(Masknp, [points], color=label_index)

        if targetsize is not None:
            Masknp = cv2.resize(Masknp, targetsize, interpolation=cv2.INTER_NEAREST)

        return Masknp

def test_single_volume(image, label, net, num_classes, patch_size=[1024, 1024]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    x, y = image.shape
    slice = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out  # 出来是二维数组,应该是1024*1024

    dice_sample, hd95_sample = 0, 0

    for i in range(1, num_classes):
        dice, hd95 = calculate_metric_percase(pred == i, label == i)
        dice_sample += dice
        hd95_sample += hd95
    dice_sample /= num_classes
    hd95_sample /= num_classes

    return dice_sample, hd95_sample, pred


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0