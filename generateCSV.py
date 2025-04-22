import os
import pandas as pd

labeled_dir = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/data/NewDataset/LabeledTrain'


aug_imageslist = []
aug_labelslist = []
# augmented_dir = os.path.join(labeled_dir, 'AugmentedTrain')
i = 0
for folder in os.listdir(labeled_dir):
    if folder in ['f', 'ori']:
        folder_path = os.path.join(labeled_dir, folder)
        images_folder = os.path.join(folder_path, 'Images')
        labels_folder = os.path.join(folder_path, 'MasksJson')
        for index, (image, label) in enumerate(zip(os.listdir(images_folder), os.listdir(labels_folder))):
            aug_imageslist.append(os.path.join(images_folder, image))
            aug_labelslist.append(os.path.join(labels_folder, label))
            i += 1
print(i)

unlabeled_dir = r'/data/cosfs/LWX/STS2024/dataset/Train-Unlabeled-20240728T073712Z-001/Train-Unlabeled'
unlabeled_images_list = os.listdir(unlabeled_dir)
unlabeled_images_list = [os.path.join(unlabeled_dir, image) for image in unlabeled_images_list]

total_imageslist = aug_imageslist + unlabeled_images_list
total_labelslist = aug_labelslist

# 确保mask_files与image_files长度相同，不足部分用None填充
total_labelslist += [None] * (len(total_imageslist) - len(total_labelslist))

# 创建一个DataFrame来存储文件路径
data = {'Images': total_imageslist, 'Masks': total_labelslist}
df = pd.DataFrame(data)
main_dir = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/data/NewDataset'
csv_filepath = os.path.join(main_dir, 'TrainDataset.csv')
df.to_csv(csv_filepath, index=False)





val_dir = r'/data/cosfs/LWX/STS2024/SSL4MIS-master/data/NewDataset/Val'
val_images_folder = os.path.join(val_dir, 'Images')
val_labels_folder = os.path.join(val_dir, 'MaskJson')
val_images_list = os.listdir(val_images_folder)
val_labels_list = os.listdir(val_labels_folder)
val_images_list = [os.path.join(val_images_folder, image) for image in val_images_list]
val_labels_list = [os.path.join(val_labels_folder, label) for label in val_labels_list]
data = {'Images': val_images_list, 'Masks': val_labels_list}
df = pd.DataFrame(data)
csv_filepath = os.path.join(main_dir, 'ValDataset.csv')
df.to_csv(csv_filepath, index=False)
