import scipy.io.wavfile
import numpy as np
import torch
import os
from dsp import mfcc


def features_extract(data_path, savepath, frame_time=0.025, hop_time=0.010):
    """ 
    提取所有对比数据集的MFCC特征并保存
    data_path:  ESC-50-master数据集位置 
    frame_time: 帧长的一段时间
    hop_time:   帧移的一段时间
    savepath:   npz数据保存路径
    """
    audio_data = []
    for audio in reversed(os.listdir(data_path)):
        if audio[0] == '5':
            break
        audio_data.append(audio)
    
    for idx, audio in enumerate(audio_data):
        audio_path = os.path.join(data_path, audio)
        sr, signal = scipy.io.wavfile.read(audio_path)
        frame_len = int(frame_time * sr)
        hop_len = int(hop_time * sr)
        norm_feature = mfcc(signal, sr, frame_len, hop_len)
        savename = os.path.join(savepath, audio[:-3]+"pt")
        torch.save({ 
                "feature": norm_feature.detach().cpu(),
                "fold": audio[0],
            },
            savename,
        )
        print(f"[{idx+1}/{len(audio_data)}]  audio: {audio}: MFCC feature save to {savename}")
    return True