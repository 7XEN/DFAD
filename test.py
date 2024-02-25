import os
import numpy as np
from keras.models import load_model
from keras.utils import image_utils

# 定义图像处理的函数
def preprocess_image(img_path, target_size=(299, 299)):
    img = image_utils.load_img(img_path, grayscale=True, target_size=target_size)
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x


# 定义获取预测结果的函数
def predict_class(model, image_array):
    predictions = model.predict(image_array)
    return np.argmax(predictions)

# 加载模型
model = load_model('test_weights/color/final/200ep.h5')
# 设置图像目录和类别标签
image_dir = 'spectrogram/A/test_color/genuine'
class_labels = ['fake', 'genuine']

# 批量处理图像并获取分类结果
for img_name in os.listdir(image_dir):
    if img_name.endswith('.png') or img_name.endswith('.jpg'):  # 假设图像是png或jpg格式
        img_path = os.path.join(image_dir, img_name)

        # 预处理图像
        processed_img = preprocess_image(img_path)

        # 获取预测类别
        predicted_class = predict_class(model, processed_img)

        # 打印结果
        print(f"Image: {img_name}, Predicted Class: {class_labels[predicted_class]}")