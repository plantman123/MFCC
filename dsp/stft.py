import numpy as np
import torch
from .fft import fft_loop


def stft(signal, frame_len, hop_len, device="cuda:0"):
    """
    signal: 1D numpy array
    frame_len: 帧长
    hop_len: 帧移
    return: spectrogram (num_frames, frame_len//2)
    """
    def hann_window(N, alpha=0.46):
        return (1-alpha) - alpha * torch.cos(2 * np.pi * torch.arange(N) / N)
    
    frame_len = int(frame_len)
    hop_len = int(hop_len)

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=complex)
    signal = signal.to(device)
    
    num_frames = int(1+(len(signal)-frame_len)//hop_len)
    window = hann_window(frame_len).to(device)
    stft_matrix = []

    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start:start + frame_len]
        frame = frame * window
        spectrum = fft_loop(frame)
        magnitude = torch.abs(spectrum[:frame_len // 2])
        stft_matrix.append(magnitude)

    return torch.stack(stft_matrix)

