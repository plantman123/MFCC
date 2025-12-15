import numpy as np


def fft_loop(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N <= 1:
        return x

    # 如果不是2的幂，回退到DFT
    if (N & (N - 1)) != 0:
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)

    log_N = int(np.log2(N))
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
        
        u_plus_t = u + t
        u_minus_t = u - t
        
        X_reshaped[:, :m2] = u_plus_t
        X_reshaped[:, m2:] = u_minus_t
        
    return X


def fft_recursive(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N == 1:
        return x
    if N % 2 != 0:
        raise ValueError("Input length must be power of 2")

    X_even = fft_recursive(x[::2])
    X_odd = fft_recursive(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    X = np.zeros(N, dtype=np.complex128)
    half = N // 2
    X[:half] = X_even + factor[:half] * X_odd
    X[half:] = X_even + factor[half:] * X_odd

    return X

