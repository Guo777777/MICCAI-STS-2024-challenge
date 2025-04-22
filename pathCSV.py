import os
import pandas as pd

# 定义图像和掩码文件夹的路径
image_folder = r'/data/cosfs/LWX/STS2024/dataset/Validation/Images'
mask_folder = r'/data/cosfs/LWX/STS2024/dataset/Validation/Masks'
# /data/cosfs/LWX/STS2024/dataset/Train-Labeled-20240728T073620Z-001/Train-Labeled/Masks
# /data/cosfs/LWX/STS2024/dataset/Validation/Images

# 获取文件夹中所有的文件名
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

# 确保图像和掩码的数量匹配
assert len(image_files) == len(mask_files), "The number of images and masks does not match."

# 创建一个DataFrame来存储文件路径
data = {'Image': image_files, 'Mask': mask_files}
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
df.to_csv('/data/cosfs/LWX/STS2024/PytorchDeepLearing-main/validation.csv', index=False)