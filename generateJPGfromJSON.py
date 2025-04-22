import math
import numpy as np
import json
import cv2
import os
import glob
import imageio

def load_label_from_json(json_path, targetsize = (53, 1024, 1024)):
    with open(json_path, 'r') as f:
        label_data = json.load(f)

    num_class = targetsize[0]
    height, width = label_data['imageHeight'], label_data['imageWidth']

    Mask_bi = np.zeros(targetsize, dtype=np.uint8)
    Mask_255 = Mask_bi.copy()

    for shape in label_data['shapes']:
        mask_bin = np.zeros((height, width), dtype=np.uint8)
        mask_255 = mask_bin.copy()

        label = shape['label']
        label_index = int(label)
        ten_digit = (int(label) // 10) % 10  # 牙齿标签的十位数
        if ten_digit <= 4:
            label_index = label_index - (11 + (ten_digit - 1) * 2)
        else:
            label_index = label_index - (19 + (ten_digit - 5) * 5)

        points = np.array(shape['points'], dtype=np.int32)
        points = points.reshape((-1, 1, 2))  # Reshape for cv2.fillPoly

        cv2.fillPoly(mask_bin, [points], 1)
        mask_bin = cv2.resize(mask_bin, (targetsize[1], targetsize[2]), interpolation=cv2.INTER_NEAREST)
        Mask_bi[label_index] = mask_bin

        cv2.fillPoly(mask_255, [points], 1)
        mask_255 = cv2.resize(mask_255, (targetsize[1], targetsize[2]), interpolation=cv2.INTER_NEAREST)
        Mask_255[label_index] = mask_255


    foreground_mask = np.max(Mask_bi[:num_class - 2, :, :], axis=0)
    Mask_bi[num_class - 1, :, :] = 1 - foreground_mask
    Mask_255[num_class - 1, :, :] = 1 - foreground_mask

    return Mask_bi, Mask_255


def find_json_files(mask_dir):
    # 使用glob来查找所有的.json文件
    json_files = glob.glob(os.path.join(mask_dir, '*.json'))
    return json_files


if __name__ == "__main__":
    mask_dir = r'D:\508\STS2024\dataset\Validation\Masks'
    save_dir = r'D:\508\STS2024\dataset\Validation\Mask_pic'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json_pathlist = find_json_files(mask_dir)
    for json_file in json_pathlist:
        mask_bi, _ = load_label_from_json(json_file)

        for i in range(mask_bi.shape[0] - 1):
            mask_bi[i, :, :] *= (i + 1)
        pred_background = mask_bi[-1, :, :]
        pred_background = np.logical_not(pred_background).astype(pred_background.dtype)
        mask_bi[-1, :, :] = pred_background
        prediction = np.sum(mask_bi, axis=0)
        prediction_uint8 = (prediction * 255).astype(np.uint8)

        json_name = os.path.basename(json_file)   # JSON文件名
        png_name = json_name.replace('.json', '.png')
        save_path = os.path.join(save_dir, png_name)
        imageio.imwrite(save_path, prediction_uint8)

