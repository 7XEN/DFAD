
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re

# 可视化库
from IPython.display import HTML as html_print
from IPython.display import display
import keras.backend as K
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess=tf.compat.v1.Session(config=config)


# 读取数据
filename = "wonderland.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = re.sub(r'[ ]+', ' ', raw_text)

# 创建字符到整数的映射
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1))

# 标准化
X = X / float(n_vocab)

# one-hot编码
y = np_utils.to_categorical(dataY)

filepath="testw_weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# 定义 LSTM 模型
model = Sequential()

model.add(CuDNNLSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.5))

model.add(CuDNNLSTM(512))
model.add(Dropout(0.5))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#预加载权重

from keras.models import load_model
#'''
filename = "testw_weights/weights-improvement-666-0.2541.hdf5"
#filename = "weights-improvement-303-0.2749_wonderland.hdf5"
model = load_model(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#'''
model.fit(X, y, epochs=1, batch_size=2048, callbacks=callbacks_list)

#第三层是输出形状为LSTM层(Batch_Size, 512)
lstm = model.layers[2]

#从中间层获取输出以可视化激活
attn_func = K.function(inputs = [model.get_input_at(0), K.learning_phase(0)],outputs = [lstm.output])


# 获取html元素
def cstr(s, color='black'):
    if s == ' ':
        return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
    else:
        return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)


# 输出html
def print_color(t):
    display(html_print(''.join([cstr(ti, color=ci) for ti, ci in t])))


# 选择合适的颜色
def get_clr(value):
    colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8'
                                                          '#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
              '#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
              '#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
    value = int((value * 100) / 5)
    return colors[value]


# sigmoid函数
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def visualize(output_values, result_list, cell_no):
	print("\nCell Number:", cell_no, "\n")
	text_colours = []
	for i in range(len(output_values)):
		text = (result_list[i], get_clr(output_values[i][cell_no]))
		text_colours.append(text)
	print_color(text_colours)

# 从随机序列中获得预测
def get_predictions(data):
	start = np.random.randint(0, len(data)-1)
	pattern = data[start]
	result_list, output_values = [], []
	print("Seed:")
	print("\"" + ''.join([int_to_char[value] for value in pattern]) + "\"")
	print("\nGenerated:")

	for i in range(1000):
		#为预测下一个字符而重塑输入数组
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)

		# 预测
		prediction = model.predict(x, verbose=0)

		# LSTM激活函数
		output = attn_func([x])[0][0]
		output = sigmoid(output)
		output_values.append(output)

		# 预测字符
		index = np.argmax(prediction)
		result = int_to_char[index]

		# 为下一个字符准备输入
		seq_in = [int_to_char[value] for value in pattern]
		pattern.append(index)
		pattern = pattern[1:len(pattern)]

		# 保存生成的字符
		result_list.append(result)
	return output_values, result_list


output_values, result_list = get_predictions(dataX)

for cell_no in [189, 435, 463]:
	visualize(output_values, result_list, cell_no)

