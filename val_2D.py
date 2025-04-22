import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

def test_single_volume(image, label, net, num_classes, patch_size=[1024, 1024]):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    x, y = image.shape
    slice = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        temp = torch.softmax(net(input).squeeze(0), dim=0)
        # temp1 = torch.softmax(temp[:-1], dim=0)
        # out = torch.cat((temp1, temp[-1:]), dim=0)
        out = temp.cpu().detach().numpy()
        pred = out

    # 预测结果二值化
    # for i in range(pred.shape[0]):
    #     arr = pred[i]
    #     min_val = np.min(arr)
    #     max_val = np.max(arr)
    #     pred[i] = (arr - min_val) / (max_val - min_val)
    #     for x in range(pred.shape[1]):
    #         for y in range(pred.shape[2]):
    #             if pred[i][x][y] > 0.5:
    #                 pred[i][x][y] = 1
    #             else:
    #                 pred[i][x][y] = 0
    for x in range(pred.shape[1]):
        for y in range(pred.shape[2]):
                max_index = np.argmax(pred[:, x, y])
                pred[max_index, x, y] = 1
                for i in range(num_classes):
                    if i != max_index:
                        pred[i, x, y] = 0

    dice_sample, hd95_sample = 0, 0

    for i in range(num_classes):
        dice, hd95 = calculate_metric_percase(pred[i], label[i])
        dice_sample += dice
        hd95_sample += hd95
    dice_sample /= num_classes
    hd95_sample /= num_classes

    return dice_sample, hd95_sample, pred

def calculate_metric_percase(pred, gt):
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if np.all(gt == 0):
            hd95 = 0
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0



def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
