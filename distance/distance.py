import torch
import numpy as np


def compute_dist(mfcc1:torch.Tensor, mfcc2:torch.Tensor, mode="DTW"):
    """
    距离计算函数，返回值越小说明两种特征越相似
    """
    dist = 100
    if mode == "DTW":
        # ?
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
        dist = distance / len(path)

    elif mode == "cosine":
        from scipy.spatial.distance import cosine
        dist = cosine(mfcc1.flatten(), mfcc2.flatten())

    elif mode == "l2":
        diff = mfcc1 - mfcc2
        if isinstance(diff, torch.Tensor):
            diff = diff.detach().cpu().numpy()
        else:
            diff = np.asarray(diff)

        if diff.ndim <= 1:
            dist = np.linalg.norm(diff)
        else:
            dist = np.linalg.norm(diff, ord='fro')
        
    else:
        print("Invalid dist-mode.")
        exit(1)

    return dist
