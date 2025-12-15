import numpy as np
from .stft import stft


def hz_mel_transform(s, reverse=False):
    """
    Mel频率与Hz频率的转换
    s: 输入频率
    reverse: 默认hz转mel
    """
    if reverse:
        return 700 * (10**(s / 2595) - 1)
    return 2595 * np.log10(1 + s / 700)


def mel_filterbank(sr, n_fft, n_mels):
    """
    创建Mel滤波器组
    sr: 采样率
    n_fft: FFT点数
    n_mels: Mel滤波器个数
    :return: 滤波器组矩阵 (n_mels, n_fft//2)
    """
    f_min = 0
    f_max = sr / 2

    mel_min = hz_mel_transform(f_min)
    mel_max = hz_mel_transform(f_max)

    # 在Mel刻度上均匀取点
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = hz_mel_transform(mel_points, True)

    # 将Hz频率转换为FFT bin索引
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filterbank = np.zeros((n_mels, n_fft // 2))

    # 构建三角滤波器
    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]
        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / (right - center)

    return filterbank


def dct(x, num_ceps):
    """
    离散余弦变换 (DCT-II)
    x: 输入信号
    num_ceps: 保留的倒谱系数个数
    :return: DCT变换结果
    """
    N = len(x)
    result = np.zeros(num_ceps)

    for k in range(num_ceps):
        for n in range(N):
            result[k] += x[n] * np.cos(np.pi * k * (2*n + 1) / (2 * N))
    return result


def mfcc(signal, sr, frame_len, hop_len, alpha_emphasis=0.9, n_mels=26, n_ceps=13):
    """
    计算MFCC特征
    signal: 输入音频信号
    sr: 采样率
    frame_len: 帧长
    hop_len: 帧移
    n_mels: Mel滤波器个数
    n_ceps: MFCC系数个数
    :return: MFCC特征矩阵
    """
    # 预加重pre-Emphasis
    # signal[i] = signal[i] - alpha * signal[i-1]
    signal = np.append(signal[0], signal[1:] - alpha_emphasis * signal[:-1])

    # stft(同时进行加窗分帧)
    spec = stft(signal, frame_len, hop_len)

    # 功率谱
    power_spec = spec ** 2

    # 应用Mel滤波器组
    fb = mel_filterbank(sr, frame_len, n_mels)
    mel_energy = np.dot(power_spec, fb.T)

    # 取对数
    mel_energy = np.where(mel_energy == 0, 1e-10, mel_energy)
    log_mel = np.log(mel_energy)

    # DCT提取MFCC特征
    mfcc_feature = np.array([dct(frame, n_ceps) for frame in log_mel])

    # 归一化
    mfcc_norm = (mfcc - np.mean(mfcc_feature, axis=0) + 1e-8)
    return mfcc_norm

