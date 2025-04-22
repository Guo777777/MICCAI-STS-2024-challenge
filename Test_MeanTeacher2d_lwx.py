import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import imageio
from dataloaders.dataset import datasetModelSegwithopencv
from networks.unet import UNet

THRESHOLD = 0.5

def pred(model_path, images_path, labeled_path, save_dir):
    # 导入模型文件
    device = torch.device("cuda:1")

    # 导入模型文件
    model = UNet(in_chns=1, class_num=53).to(device)  # 直接在创建模型时移动到device
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 准备数据集
    db_test = datasetModelSegwithopencv(images_path, labeled_path, split="val")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # 预测
    for i_batch, sampled_batch in enumerate(test_loader):
        image = sampled_batch["image"].squeeze(0).cpu().detach().numpy()
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
        model.eval()
        # with torch.no_grad():
        #     temp = torch.softmax(model(input).squeeze(0), dim=0)
        #     # temp1 = torch.softmax(temp[:-1], dim=0)
        #     # out = torch.cat((temp1, temp[-1:]), dim=0)
        #     out = temp.cpu().detach().numpy()
        #     pred = out
        # # 预测结果二值化
        # for i in range(pred.shape[0]):
        #     arr = pred[i]
        #     min_val = np.min(arr)
        #     max_val = np.max(arr)
        #     pred[i] = (arr - min_val) / (max_val - min_val)
        #     for x in range(pred.shape[1]):
        #         for y in range(pred.shape[2]):
        #             if pred[i][x][y] > THRESHOLD:
        #                 pred[i][x][y] = 1
        #             else:
        #                 pred[i][x][y] = 0

        with torch.no_grad():
            temp = torch.softmax(model(input).squeeze(0), dim=0)
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
                for i in range(53):
                    if i != max_index:
                        pred[i, x, y] = 0

        # for x in range(pred.shape[1]):
        #     for y in range(pred.shape[2]):
        #         if pred[-1:, x, y] > THRESHOLD:
        #             pred[-1:, x, y] = 1
        #             pred[:-1, x, y] = 0
        #         else:
        #             max_index = np.argmax(pred[:-1, x, y])
        #             pred[max_index, x, y] = 1
        #             pred[:max_index, x, y] = 0
        #             pred[max_index+1:-1, x, y] = 0
                # max_index = np.argmax(pred[:, x, y])
                # pred[max_index, x, y] = 1
                # pred[:max_index, x, y] = 0
                # pred[max_index+1:, x, y] = 0

        height, width = pred.shape[1], pred.shape[2]
        Masknp = pred

        for label_index in range(1, Masknp.shape[0]):
            masknp = np.zeros((height, width), dtype=np.uint8)
            masknp[pred[label_index] == 1] = label_index  # 为每个标签设置不同的灰度值
            Masknp[label_index] = masknp

        # for i in range(1, Masknp.shape[0]):
        #     Masknp[i, :, :] *= i

        foreground_mask = np.max(Masknp[1:Masknp.shape[0], :, :], axis=0)
        Masknp[0, :, :] = 1 - foreground_mask

        # for i in range(Masknp.shape[0] - 1):
        #     Masknp[i, :, :] *= (i + 1)
        pred_background = Masknp[0, :, :]
        pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
        Masknp[0, :, :] = pred_background
        prediction = np.sum(Masknp, axis=0)

        prediction_uint8 = (prediction * 255).astype(np.uint8)


        # 保存图像
        ori_mask_name = os.path.basename(images_path[i_batch])
        mask_name = ori_mask_name.replace('.jpg', '_mask.jpg')
        mask_path = os.path.join(save_dir, mask_name)
        imageio.imwrite(mask_path, prediction_uint8)

        print('{} 已保存'.format(mask_name))


    print('Prediction finish!')


if __name__ == '__main__':
    datafile_dir = r'/data/cosfs/LWX/STS2024/dataset/Validation-Public-20240728T073621Z-001/validation.csv'
    csv_data = pd.read_csv(datafile_dir)
    test_images = csv_data.iloc[:, 0].values
    test_labels = csv_data.iloc[:, 1].values
    model_path = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/model_log/20240824_152217/unet_best_model.pth'
    model_filename = os.path.basename(os.path.dirname(model_path)) + '/' + os.path.splitext(os.path.basename(model_path))[0]
    save_dir = r'/data/cosfs/LWX/STS2024/dataset/Validation-Public-20240728T073621Z-001/{}/{}'.format(model_filename, THRESHOLD)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pred(model_path, test_images, test_labels, save_dir)