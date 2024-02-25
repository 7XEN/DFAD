import cv2
import os


def resize_images(input_folder, output_folder, new_size):
    # 获取输入文件夹中的所有图像文件名
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # 遍历每张图像并调整其大小
    for file in image_files:
        input_image_path = os.path.join(input_folder, file)

        # 读取原始图像
        img = cv2.imread(input_image_path)

        # 设置新的图像大小
        resized_img = cv2.resize(img, (new_size[0], new_size[1]))

        # 保存修改后的图像到指定目录
        output_image_path = os.path.join(output_folder, file)
        cv2.imwrite(output_image_path, resized_img)


# 设置输入文件夹路径、输出文件夹路径以及新的图像大小（宽度x高度）
input_folder = "spectrogram/A/original/fake_color"
output_folder = "spectrogram/A/299/color/fake"
new_size = (299, 299)

# 调用函数来处理图像
resize_images(input_folder, output_folder, new_size)