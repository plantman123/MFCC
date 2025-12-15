import numpy as np
import scipy.io.wavfile
import os
from dsp import mfcc 


def features_extract(data_path, frame_time=0.025, hop_time=0.010, savepath="./features"):
    """ 
    提取所有对比数据集的MFCC特征并保存
    data_path:  ESC-50-master数据集位置 
    frame_time: 帧长的一段时间
    hop_time:   帧移的一段时间
    savepath:   npy矩阵保存路径
    """
    audio_data = []
    for audio in reversed(os.listdir(data_path)):
        if audio[0] == '5':
            break
        audio_data.append(audio)

    for audio in audio_data:
        audio_path = os.path.join(data_path, audio)
        sr, signal = scipy.io.wavfile.read(audio_path)
        frame_len = int(frame_time * sr)
        hop_len = int(hop_time * sr)
        feature = mfcc(signal, sr, frame_len, hop_len)
        norm_feature = (feature - np.mean(feature, axis=0) + 1e-8)
        np.save(os.path.join(savepath, audio[:-3]+"npy"), norm_feature)
        print(f"audio: {audio}, MFCC feature computation finished, save to {os.path.join(savepath, audio[:-3]+'npy')}")
    return True


def main():
    features_extract("./data/ESC-50-master/audio")


if __name__ == "__main__":
    # print(fft_recursive(np.array([1, 2, 3, 4])))
    # print(fft_loop(np.array([1, 2, 3, 4])))
    main()
