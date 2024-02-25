from pydub import AudioSegment
import os

# 设置FLAC音频文件目录和输出WAV目录
flac_directory = 'audio/fake'
wav_directory = 'audio/fake_wav'

# 确保输出目录存在
if not os.path.exists(wav_directory):
    os.makedirs(wav_directory)

# 遍历FLAC目录中的所有文件
for filename in os.listdir(flac_directory):
    if filename.endswith(('.flac',)):
        # 读取FLAC文件
        flac_path = os.path.join(flac_directory, filename)
        audio = AudioSegment.from_file(flac_path, format="flac")

        # 生成输出WAV文件的路径和名称
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        wav_path = os.path.join(wav_directory, wav_filename)

        # 将FLAC转换为WAV并保存
        audio.export(wav_path, format="wav")

        print(f"Converted {flac_path} to {wav_path}")

print("All FLAC files have been converted to WAV format.")