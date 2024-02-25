import cv2
import os

# 设置输入文件夹路径和输出文件夹路径
input_folder = "spectrogram/A/original/fake_color"
output_folder = "spectrogram/A/original/fake_color"

# 获取输入文件夹中的所有图像文件名
image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

for image_file in image_files:
    # 构建完整的输入文件路径和输出文件路径
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)

    # 加载图像并定义要裁切的区域（左上角点、右下角点）
    img = cv2.imread(input_path)
    x1, y1 = (150, 400)  # 左上角点坐标
    x2, y2 = (890, 700)  # 右下角点坐标

    # 根据指定的区域对图像进行裁切
    cropped_img = img[y1:y2, x1:x2]

    # 保存裁切后的图像到输出文件夹
    cv2.imwrite(output_path, cropped_img)