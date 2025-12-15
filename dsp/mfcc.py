import numpy as np
from stft import stft

def mel_hz_transform(s, reverse=False):
    if reverse:
        return 2595 * np.log10(1 + s / 700)
    return 700 * (10**(s / 2595) - 1)


def mel_filterbank(sr, n_fft, n_mels):
    """
    mel滤波器
    """
    f_min = 0
    f_max = sr / 2

    mel_min = mel_hz_transform(f_min, True)
    mel_max = mel_hz_transform(f_max, True)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_hz_transform(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2))

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
    DCT-II
    """
    N = len(x)
    result = np.zeros(num_ceps)

    for k in range(num_ceps):
        for n in range(N):
            result[k] += x[n] * np.cos(np.pi * k * (2*n + 1) / (2 * N))
    return result


def mfcc(signal, sr, frame_len, hop_len, n_mels=26, n_ceps=13):
    spec = stft(signal, frame_len, hop_len)
    power_spec = spec ** 2

    fb = mel_filterbank(sr, frame_len, n_mels)
    mel_energy = np.dot(power_spec, fb.T)

    mel_energy = np.where(mel_energy == 0, 1e-10, mel_energy)
    log_mel = np.log(mel_energy)

    mfcc_feat = np.array([dct(frame, n_ceps) for frame in log_mel])

    return mfcc_feat

