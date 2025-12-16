import torch
import numpy as np


def fft_loop(x, device="cuda:0"):
    """
    fft循环实现，使用gpu加速
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x = x.to(torch.complex64)
    x = x.to(device) 
    if x.dim() == 1:
        x = x.unsqueeze(0)
    
    batch_size, N = x.shape

    if N <= 1:
        return x.squeeze(0)
    
    # 回退DFT
    if (N & (N - 1)) != 0:
        n = torch.arange(N, device=device, dtype=torch.float32)
        k = n.view(N, 1)
        M = torch.exp(-2j * np.pi * k * n / N)    
        return torch.matmul(x, M).squeeze(0)

    log_N = int(np.log2(N))
    n = torch.arange(N, device=device, dtype=torch.long)
    rev = torch.zeros(N, device=device, dtype=torch.long)
    for i in range(log_N):
        rev |= ((n >> i) & 1) << (log_N - 1 - i)
    
    X = x[:, rev]
    for s in range(1, log_N + 1):
        m = 1 << s
        m2 = m >> 1
        w = torch.exp(-2j * torch.pi * torch.arange(m2, device=device) / m)
        X_reshaped = X.view(batch_size, -1, m)
        
        u = X_reshaped[:, :, :m2]
        t = w * X_reshaped[:, :, m2:]
        
        u_plus_t = u + t
        u_minus_t = u - t
        X_new = torch.cat([u_plus_t, u_minus_t], dim=2)
        X = X_new.view(batch_size, N)
        
    return X.squeeze(0)


def fft_recursive(x, device="cuda:0"):
    x = torch.asarray(x, dtype=torch.complex128)
    N = x.shape[0]
    if N == 1:
        return x
    if (N & (N - 1)) != 0:
        n = torch.arange(N, device=device, dtype=torch.float32)
        k = n.view(N, 1)
        M = torch.exp(-2j * np.pi * k * n / N)
        return torch.matmul(x, M).squeeze(0)

    X_even = fft_recursive(x[::2])
    X_odd = fft_recursive(x[1::2])

    factor = torch.exp(-2j * torch.pi * torch.arange(N) / N)

    X = torch.zeros(N, dtype=torch.complex128)
    half = N // 2
    X[:half] = X_even + factor[:half] * X_odd
    X[half:] = X_even + factor[half:] * X_odd

    return X

