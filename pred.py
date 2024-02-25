import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import os


model = tf.keras.models.load_model('test_weights/color/final/train.h5')
sub_dirs = 'spectrogram/A/test_color/fake'
pred_samp = 'fake'
pred_cls = []
classcatalogue = ['fake', 'genuine']


#img_path = 'spectrogram/B/colour/256/fake/7_fake.wav.png'
string_list = []
sub_dir = [x[2] for x in os.walk(sub_dirs)]
ct1 = 0
ct2 = 0
for sub in sub_dir[0]:
    #image = cv2.imread(sub_dirs + '\\' + sub)
    #img = image
    img = image.load_img(sub_dirs + '\\' + sub, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)

    print(predictions)

    predicted_class = np.argmax(predictions, axis=1)
    pred_cls = classcatalogue[int(predicted_class)]
    if pred_cls == pred_samp:
        ct1 += 1
    print("识别为:", classcatalogue[int(predicted_class)])
    ct2 += 1
    string_list.append(classcatalogue[int(predicted_class)])
print(string_list)
accu = ct1 / ct2
print('准确率为:',accu)


