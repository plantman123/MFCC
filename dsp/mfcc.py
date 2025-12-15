import numpy as np
import torch
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
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = hz_mel_transform(mel_points, True)

    # 将Hz频率转换为FFT bin索引
    bins = torch.floor((n_fft + 1) * hz_points / sr).long()
    filterbank = torch.zeros((n_mels, int((n_fft // 2))))

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
    离散余弦变换
    x: 输入信号 (Tensor) [..., N]
    num_ceps: 保留的倒谱系数个数
    :return: DCT变换结果 [..., num_ceps]
    """
    N = x.shape[-1]
    k = torch.arange(num_ceps, device=x.device, dtype=x.dtype).unsqueeze(1)
    n = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0)
    dct_matrix = torch.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return torch.matmul(x, dct_matrix.T)


def mfcc(signal, sr, frame_len, hop_len, alpha_emphasis=0.9, n_mels=26, n_ceps=13, device="cuda:0"):
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

    # 数据预处理
    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal)
    # 注意：.to() 不是原地操作，必须赋值回去
    signal = signal.to(device)

    # stft(同时进行加窗分帧)
    # 传递 device 参数给 stft
    spec = stft(signal, frame_len, hop_len, device=device)

    # 功率谱
    power_spec = spec ** 2

    # 应用Mel滤波器组
    fb = mel_filterbank(sr, frame_len, n_mels).to(device)
    mel_energy = torch.matmul(power_spec, fb.T)

    # 取对数
    mel_energy = torch.where(mel_energy == 0, torch.tensor(1e-10, device=device), mel_energy)
    log_mel = torch.log(mel_energy)

    # DCT提取MFCC特征
    # 直接对整个 batch 进行 DCT
    mfcc_feature = dct(log_mel, n_ceps)

    # 归一化
    mfcc_norm = (mfcc_feature - torch.mean(mfcc_feature, dim=0) + 1e-8)
    return mfcc_norm

