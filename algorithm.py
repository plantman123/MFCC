import numpy as np


def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N <= 1:
        return x

    log_N = int(np.log2(N))
    if 1 << log_N != N:
        raise ValueError("Input length must be power of 2")

    n = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(log_N):
        rev |= ((n >> i) & 1) << (log_N - 1 - i)
    
    X = x[rev]

    for s in range(1, log_N + 1):
        m = 1 << s
        m2 = m >> 1
        w = np.exp(-2j * np.pi * np.arange(m2) / m)
        X_reshaped = X.reshape(-1, m)
        u = X_reshaped[:, :m2]
        t = w * X_reshaped[:, m2:]
        X_reshaped[:, :m2] = u + t
        X_reshaped[:, m2:] = u - t
        
    return X
