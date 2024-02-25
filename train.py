import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import os
import xlwt
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Conv2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping


training_images = np.array(np.load('spectrogram/A/299/color/training_images.npy', allow_pickle=True))
training_labels = np.array(np.load('spectrogram/A/299/color/training_labels.npy', allow_pickle=True))
validation_images = np.array(np.load('spectrogram/A/299/color/validation_images.npy', allow_pickle=True))
validation_labels = np.array(np.load('spectrogram/A/299/color/validation_labels.npy', allow_pickle=True))
testing_images = np.array(np.load('spectrogram/A/299/color/testing_images.npy', allow_pickle=True))
testing_labels = np.array(np.load('spectrogram/A/299/color/testing_labels.npy', allow_pickle=True))


print(testing_images.shape)


if __name__ == '__main__':

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(299, 299, 3), data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax'))

    model.summary()

    sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.2, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
    #CNN
    x_train = training_images
    x_test = testing_images
    y_train = np_utils.to_categorical(training_labels, 2)
    y_test = np_utils.to_categorical(testing_labels, 2)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    dir_path = "test_weights/color/test.h5"
    checkpoint = ModelCheckpoint(dir_path, monitor='val_acc', verbose=0, save_weights_only=True, save_best_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=100)


    result_train = model.evaluate(x_train, y_train)
    print('\nTrain CNN Acc:\n', result_train[1])

    result_test = model.evaluate(x_test, y_test)
    print('\nTest CNN Acc:\n', result_test[1])

    model.save('test_weights/color/final/train.h5')

'''
    model2 = Sequential()                                     #下方color_channel
    model2.add(Conv2D(25, (3, 3), strides=1, input_shape=(299, 299, 1), data_format='channels_last'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Conv2D(25, (3, 3), strides=1))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D((2, 2)))

    model2.add(Flatten())

    model2.add(Dense(units=100,activation='relu'))

    model2.add(Dense(units=2, activation='softmax'))

    model2.summary()

    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model2.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])


    checkpoint = ModelCheckpoint('testw_weights/model.h5', monitor='val_accuracy', save_best_only=True)

    #CNN
    x_train = training_images
    x_test = testing_images
    y_train = np_utils.to_categorical(training_labels, 2)
    y_test = np_utils.to_categorical(testing_labels, 2)

    model2.fit(x_train, y_train, batch_size=32, epochs=200)
    
    # 打印在训练集上的训练结果
    result_train = model2.evaluate(x_train, y_train)
    print('\nTrain CNN Acc:\n', result_train[1])
    # 打印在测试集上的测试结果
    result_test = model2.evaluate(x_test, y_test)
    print('\nTest CNN Acc:\n', result_test[1])
    # 保存模型为.h5，方便下次预测
    model2.save('testw_weights/1_final/train.h5')
'''