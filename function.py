from collections import defaultdict
import scipy.io.wavfile
import pandas as pd
import numpy as np
import torch
import os
from dsp import mfcc
from distance import compute_dist
import multiprocessing


_global_audio_features = None

def _init_worker(audio_features):
    global _global_audio_features
    _global_audio_features = audio_features

def _compute_score_worker(idx, test_feature, dist_mode):
    target_dct = _global_audio_features[idx]
    audio_target = target_dct.get("target")
    audio_feature = _select_feature_representation(target_dct, dist_mode)
    return (audio_target, compute_dist(audio_feature, test_feature, dist_mode))


def _select_feature_representation(sample_dct, dist_mode):
    if dist_mode == "DTW":
        return sample_dct.get("feature")
    agg_feature = sample_dct.get("agg_feature")
    if agg_feature is None:
        feature = sample_dct.get("feature")
        if feature is None:
            return None
        mean = torch.mean(feature, dim=0)
        std = torch.std(feature, dim=0)
        agg_feature = torch.cat([mean, std], dim=0)
        sample_dct["agg_feature"] = agg_feature
    return agg_feature


def get_target(filename:str):
    """
    从文件名中提取对应的类别分类target
    """
    idx = filename.find('.')
    for i in reversed(range(0, idx)):
        if filename[i] == '-':
            target = filename[i+1: idx]
            return target
    return None


def load_esc50(datapath:str="./data/ESC-50-master/meta/esc50.csv"):
    """
    加载数据并划分
    fold1-4为数据库，fold5为需要分类的数据
    """
    all_data = pd.read_csv(datapath)
    filenames = all_data.get("filename")
    audio_data = []
    test_data = []
    for filename in filenames:
        if filename[0] == '5':
            test_data.append(filename[:-4]+".pt")
        else:
            audio_data.append(filename[:-4]+".pt")
    
    # audio_data、test_data后缀为.pt，filenames后缀为.wav
    return audio_data, test_data, filenames
    

def features_extract(audio_data:list, savepath:str, frame_time=0.025, hop_time=0.010, n_mels=40, n_ceps=20, lifter=22, norm='cms', data_path="./data/ESC-50-master/audio"):
    """ 
    提取所有对比数据集的MFCC特征并保存
    audio_data: 所有需要进行mfcc特征提取的文件名list
    data_path:  ESC-50-master数据集位置 
    frame_time: 帧长的一段时间
    hop_time:   帧移的一段时间
    savepath:   npz数据保存路径
    """
    for idx, audio in enumerate(audio_data):
        audio_path = os.path.join(data_path, audio)
        sr, signal = scipy.io.wavfile.read(audio_path)
        frame_len = int(frame_time * sr)
        hop_len = int(hop_time * sr)
        norm_feature, agg_feature = mfcc(
            signal,
            sr,
            frame_len,
            hop_len,
            n_mels=n_mels,
            n_ceps=n_ceps,
            lifter_coeff=lifter,
            norm=norm,
            return_agg=True,
        )
        savename = os.path.join(savepath, audio[:-4]+".pt")
        torch.save({ 
                "feature": norm_feature.detach().cpu(),
                "agg_feature": agg_feature.detach().cpu(),
                "filename": audio[:-4]+".pt",
                "fold": audio[0],
                "target": get_target(audio),
            },
            savename,
        )
        print(f"[{idx+1}/{len(audio_data)}] audio: {audio}: MFCC feature save to {savename}")
    return True


def classification(score_list:list, alpha:float=2.8, beta:float=3.4, lambd:float=0.77, eps:float=1e-8):
    """
    分类函数：根据得到的结果进行最后的分类
    score_list: 结果列表，每一项为一个元组，第一项为target，第二项为dis-score
    其余参数为超参，最优超参有optimize-hyperparameters获取
    """
    min_dist = min(score_list, key=lambda x: x[1])[1]
    norm_list = [(t, d / (min_dist + eps)) for t, d in score_list]
    sum_term = defaultdict(float)
    min_term = defaultdict(lambda: float("inf"))

    for target, dist in norm_list:
        sum_term[target] += np.exp(-alpha * dist)
        if dist < min_term[target]:
            min_term[target] = dist
    score_dct = {}
    for target in sum_term:
        score_dct[target] = (
            sum_term[target]
            + lambd * np.exp(-beta * min_term[target])
        )
    pred_class = max(score_dct.items(), key=lambda x: x[1])[0]
    return pred_class, score_dct


def evaluate(test_data:list, audio_data:list, frame_time:float, hop_time:float, k:int=5, dist_mode="DTW", threads=128, test_mode="hit", n_mels=40, n_ceps=20, lifter=22, norm='cms'):
    """
    评估函数，使用已经计算好的特征获取fold5中对应的target，与真实target对比打分
    test_data:  测试文件名lis
    audio_data: 数据库文件名list
    frame_time: 帧长的一段时间
    hop_time:   帧移的一段时间
    k:          最终判断正误的范围
    dist_mode:   相似度函数
    threads:    并行计算线程数
    """
    # 加载保存的mfcc特征并保存
    audio_features = []
    for audio_filename in audio_data:
        audio_feature_path = os.path.join(
            f"./features/frame_time={frame_time}_hop_time={hop_time}_n_mels={n_mels}_n_ceps={n_ceps}_lifter={lifter}_norm={norm}", 
            audio_filename)
        audio_features.append(torch.load(audio_feature_path, map_location="cpu"))

    # 加载需要测试分类的mmfcc特征
    topk_dct = {}
    all_cnt = 0
    right_cnt = 0
    num_workers = min(multiprocessing.cpu_count(), threads)
    
    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(audio_features,)) as pool:
        for idx, test_filename in enumerate(test_data):
            topk_dct[test_filename] = []
            test_feature_path = os.path.join(
                f"./features/frame_time={frame_time}_hop_time={hop_time}_n_mels={n_mels}_n_ceps={n_ceps}_lifter={lifter}_norm={norm}", 
                test_filename)
            test_feature_dct = torch.load(test_feature_path,  map_location="cpu")
            test_target = test_feature_dct.get("target")

            if dist_mode == "DTW":
                # DTW情况：先使用l2粗排再使用DTW精排
                test_feature = _select_feature_representation(test_feature_dct, "l2")
                args = [(i, test_feature, "l2") for i in range(len(audio_features))]
                score_list = pool.starmap(_compute_score_worker, args)
                score_list_with_idx = list(enumerate(score_list))
                top50_candidates = sorted(score_list_with_idx, key=lambda x: x[1][1])[:30]
                top50_indices = [x[0] for x in top50_candidates]

                # DTW精排
                test_feature = _select_feature_representation(test_feature_dct, "DTW")
                args = [(i, test_feature, "DTW") for i in top50_indices]
                score_list = pool.starmap(_compute_score_worker, args)
                score_list = sorted(score_list, key=lambda x: x[1])[:k]
            
            else:
                # 其他情况：直接使用对应的距离函数排序
                test_feature = _select_feature_representation(test_feature_dct, dist_mode)
                # 最终的相似度得分list，每一项为一个元组，第一项为traget，第二项为相似度score
                args = [(i, test_feature, dist_mode) for i in range(len(audio_features))]
                score_list = pool.starmap(_compute_score_worker, args)
                score_list = sorted(score_list, key=lambda x: x[1])[:k]
            
            if test_mode == "score":
                pred_target, _ = classification(score_list)
                if pred_target == test_target:
                    right_cnt += 1

            elif test_mode == "hit":
                for item in score_list:
                    if item[0] == test_target:
                        right_cnt += 1
                        break
            
            else:
                print("Invalid evaluate mode.")
                exit(1)

            all_cnt += 1
            print(f"[{idx+1}/{len(test_data)}] test finish, testfile={test_feature_dct.get('filename')}")
            print(f"now accuracy is {right_cnt/all_cnt}")

    return right_cnt / all_cnt

