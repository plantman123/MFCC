import argparse
import os
from function import (
    load_esc50,
    features_extract,
    evaluate,
)


def main(args):
    savepath = os.path.join("./features", f"frame_time={args.frame_time}_hop_time={args.hop_time}")
    audio_data, test_data, all_data = load_esc50()

    # 特征提取存储(如果还没有保存)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        features_extract(
            audio_data=all_data, 
            savepath=savepath,
            frame_time=args.frame_time,
            hop_time=args.hop_time,
        )

    # 相似mfcc特征阵计算获取
    if len(os.listdir(savepath))==2000:
        evaluate(
            test_data, 
            audio_data, 
            args.frame_time, 
            args.hop_time, 
            k=5, 
            sim_mode=args.sim_mode,
        )   


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/ESC-50-master/audio")
    parser.add_argument("--frame_time", type=float, default=0.025)
    parser.add_argument("--hop_time", type=float, default=0.010)
    parser.add_argument("--sim_mode", type=str, default="DTW", help="sim mode")
    args = parser.parse_args()
    main(args)

