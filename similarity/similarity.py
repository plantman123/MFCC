import torch


def compute_sim(mfcc1:torch.Tensor, mfcc2:torch.Tensor, mode="DTW"):
    """
    相似度计算函数，返回值越小说明两种特征越相似（距离）
    """
    sim = None
    if mode == "DTW":
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
        sim = distance

    elif mode == "cosine":
        from scipy.spatial.distance import cosine
        sim = cosine(mfcc1.flatten(), mfcc2.flatten())

    elif mode == "l2":
        import numpy as np
        diff = mfcc1 - mfcc2
        if isinstance(diff, torch.Tensor):
            diff = diff.detach().cpu().numpy()
        else:
            diff = np.asarray(diff)

        if diff.ndim <= 1:
            dist = np.linalg.norm(diff)
        else:
            dist = np.linalg.norm(diff, ord='fro')
        sim = dist
        
    else:
        print("Invalid sim mode.")
        exit(1)

    return sim
