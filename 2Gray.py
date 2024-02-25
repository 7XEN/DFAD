'''
import cv2

filename=('spectrogram/1.wav.png')

# 读取彩色图像文件
img_color = cv2.imread(filename)

# 转换为灰度图像
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 保存灰度图像文件
cv2.imwrite(filename, img_gray)
'''
from PIL import Image
import os

# 指定彩色图像文件夹路径和灰度图像文件夹路径
color_folder = "spectrogram/A/299/fake"
gray_folder = "spectrogram/A/299/fake"

# 如果灰度图像文件夹不存在，则创建
if not os.path.exists(gray_folder):
    os.mkdir(gray_folder)

# 遍历彩色图像文件夹中的所有文件
for filename in os.listdir(color_folder):
    # 打开彩色图像文件
    img_color = Image.open(os.path.join(color_folder, filename))

    # 转换为灰度图像
    img_gray = img_color.convert("L")

    # 保存灰度图像文件
    img_gray.save(os.path.join(gray_folder, filename))