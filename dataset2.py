import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
# 设置图片数据集的目录
dataset_directory = 'spectrogram/A/299/color'

# 初始化存储图片的列表
all_images = []
all_labels = []

# 遍历数据集的每个子文件夹（即每个类别）
for category_index, category in enumerate(os.listdir(dataset_directory)):
    category_path = os.path.join(dataset_directory, category)

    # 确保它是一个目录
    if os.path.isdir(category_path):
        # 遍历该类别的所有图片文件
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)

            # 确保它是一个图片文件
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 加载图片并转换为numpy数组
                with Image.open(image_path) as img:


                    img_array = np.array(img)

                    # 将图片数据和对应的类别标签添加到列表中
                    all_images.append(img_array)
                    all_labels.append(category_index)

                # 将图片数据和标签转换为NumPy数组
all_images_array = np.array(all_images)
all_labels_array = np.array(all_labels)

# 分割数据集为训练集、测试集和验证集
# 假设我们想要80%的数据用于训练，10%用于验证，10%用于测试
X_train, X_temp, y_train, y_temp = train_test_split(all_images_array, all_labels_array, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 保存数据集为.npy文件
np.save(os.path.join(dataset_directory, 'training_images.npy'), X_train)
np.save(os.path.join(dataset_directory, 'training_labels.npy'), y_train)
np.save(os.path.join(dataset_directory, 'validation_images.npy'), X_val)
np.save(os.path.join(dataset_directory, 'validation_labels.npy'), y_val)
np.save(os.path.join(dataset_directory, 'testing_images.npy'), X_test)
np.save(os.path.join(dataset_directory, 'testing_labels.npy'), y_test)

print("图片数据集已分类并转为训练、验证和测试集的.npy文件。")