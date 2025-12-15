import numpy as np
from fft import fft_loop


def stft(signal, frame_len, hop_len):
    """
    signal: 1D numpy array
    frame_len: 帧长
    hop_len: 帧移
    return: spectrogram (num_frames, frame_len//2)
    """
    def hann_window(N):
        return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)
    
    window = hann_window(frame_len)
    num_frames = 1 + (len(signal) - frame_len) // hop_len

    stft_matrix = []

    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start:start + frame_len]
        frame = frame * window

        spectrum = fft_loop(frame)
        magnitude = np.abs(spectrum[:frame_len // 2])

        stft_matrix.append(magnitude)

    return np.array(stft_matrix)