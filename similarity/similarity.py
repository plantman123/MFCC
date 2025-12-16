import torch


def compute_sim(mfcc1:torch.Tensor, mfcc2:torch.Tensor, mode="DTW"):
    sim = None
    if mode == "DTW":
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)
        sim = distance

    elif mode == "cosine":
        from scipy.spatial.distance import cosine
        sim = 1 - cosine(mfcc1.flatten(), mfcc2.flatten())

    elif mode == "l2":
        import numpy as np
        dist = np.linalg.norm(np.array(mfcc1-mfcc2), ord='fro')
        sim = dist
    else:
        print("Invalid sim mode.")
        exit(1)
    return sim
