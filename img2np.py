from PIL import Image
import numpy as np
import cv2
import os

os.system("taskset -p 0xff %d" % os.getpid())

input_folder = "spectrogram/fake_gray/"
output_folder = "spectrogram/np_fake/"
# 获取输入文件夹中的所有图像文件名
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
for image_file in image_files:
    # 构建完整的输入文件路径和输出文件路径
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
# 读取图像文件
    img = cv2.imread(input_path)

# 将图像转换为NumPy数组
    array = np.array(img)

# 保存NumPy数组到文件
    np.save(output_path, array)