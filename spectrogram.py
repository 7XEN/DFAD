
import os
import torch
import torchaudio
import torchvision.transforms.functional as F
from PIL import Image
from glob import glob
import numpy as np



import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from matplotlib.colors import LogNorm





#24位
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

# 设置音频文件目录
audio_directory = 'audio/test_genuine'

# 遍历目录中的所有音频文件
for filename in os.listdir(audio_directory):
    if filename.endswith(('.wav')):  # 根据你的音频文件格式进行调整
        # 读取音频文件
        audio_path = os.path.join(audio_directory, filename)
        sample_rate, data = read(audio_path)

        # 使用matplotlib的specgram函数绘制频谱图
        plt.figure(figsize=(12, 8))
        plt.specgram(data, Fs=sample_rate, NFFT=2048, noverlap=1024, cmap='viridis', mode='default')
        plt.title(f'Spectrogram of {filename}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        # 显示颜色条
        plt.colorbar(label='Amplitude')

        # 保存图像
        plt.savefig(f'spectrogram/A/original/genuine_color/spectrogram_{filename.replace(".", "_")}.png')

        # 可选：显示图像
        # plt.show()

        # 关闭图形以避免内存泄漏
        plt.close()
'''



'''
#8位
audio_directory = 'audio/genuine'
output_directory = 'spectrogram/A/original/genuine'

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取音频文件列表
audio_files = glob(os.path.join(audio_directory, '*.flac'))


# 定义一个函数来将tensor转换为PIL图像
def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(min=-10, max=10)  # 可以根据需要调整clamp的值
    tensor = (tensor + 10) / 20 * 255  # 将tensor范围从[-10, 10]映射到[0, 255]
    tensor = tensor.permute(1, 0).numpy().astype(np.uint8)  # 转换维度并转换为uint8
    image = Image.fromarray(tensor)
    return image


# 批量处理音频文件并生成频谱图
for audio_file in audio_files:
    # 读取音频文件
    waveform, sample_rate = torchaudio.load(audio_file)

    # 确保waveform是二维的（对于单通道音频）
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

        # 计算频谱图
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=1024,  # FFT窗口大小
        win_length=512,  # 窗口长度
        hop_length=64,  # 窗口之间的跳跃长度



    )(waveform)
    spectrogram = torch.log(spectrogram + torch.finfo(torch.float32).eps)  # 取对数并添加一个小的epsilon来避免log(0)


    # 转换频谱图为图像
    spectrogram_image = tensor_to_pil(spectrogram[0, :, :])  # 选择第一个通道（对于单通道音频）

    # 保存频谱图到文件
    output_file = os.path.join(output_directory, os.path.basename(audio_file).replace('.flac', '.png'))
    spectrogram_image.save(output_file)

print("Spectrogram generation complete.")






'''

import os
import torch
import torchaudio
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
from glob import glob

# 设置音频文件的目录和输出频谱图的目录
audio_directory = 'audio/test'
output_directory = 'spectrogram/A/'

# 确保输出目录存在
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取音频文件列表
audio_files = glob(os.path.join(audio_directory, '*.flac'))  # 假设音频文件是.wav格式

# 初始化Spectrogram变换
spectrogram_transform = Spectrogram(
    #sample_rate=16000,  # 假设采样率为16kHz，根据您的音频调整
    n_fft=2048,  # FFT窗口大小
    win_length=1024,  # 窗口长度
    hop_length=512,  # 窗口之间的跳跃长度
    pad_mode='reflect',  # 填充模式
    power=2.0,  # 对幅度进行幂运算以强调更强的频率分量
    normalized=True,  # 是否将频谱图归一化
    onesided=True,  # 是否只计算单边的频谱图（忽略负频率）
)

# 批量处理音频文件并生成频谱图
for audio_file in audio_files:
    # 读取音频文件
    waveform, sample_rate = torchaudio.load(audio_file)

    # 计算频谱图
    spectrogram = spectrogram_transform(waveform).squeeze(0)

    # 将频谱图转换为分贝值
    spectrogram_db = torch.amplitude_to_db(spectrogram, top_db=80)

    # 使用matplotlib绘制彩色频谱图
    plt.figure(figsize=(12, 8))
    plt.imshow(spectrogram_db, aspect='auto', cmap='viridis')  # 使用'viridis'或其他colormap
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')

    # 保存频谱图到文件
    output_file = os.path.join(output_directory, os.path.basename(audio_file).replace('.flac', '.png'))
    plt.savefig(output_file)

    # 关闭当前的matplotlib图像，以释放内存
    plt.close()

print("Spectrogram generation complete.")
'''