import wave

# 打开WAV文件
with wave.open('audio/add_genuine/1.wav', 'rb') as wf:
#with wave.open('audio/add_fake/LJ001-0001_gen.wav', 'rb') as wf:
    # 获取声道数量
    num_channels = wf.getnchannels()
    print(f"声道数量: {num_channels}")

    # 你还可以获取其他信息，例如：
    # 采样宽度（每个样本的字节数）
    sample_width = wf.getsampwidth()
    print(f"采样宽度: {sample_width}")

    # 帧率（每秒的样本数）
    framerate = wf.getframerate()
    print(f"帧率: {framerate}")

    # 总的样本数量
    nframes = wf.getnframes()
    print(f"总样本数: {nframes}")

    # 压缩类型
    comptype = wf.getcomptype()
    print(f"压缩类型: {comptype}")

    # 压缩名称
    compname = wf.getcompname()
    print(f"压缩名称: {compname}")