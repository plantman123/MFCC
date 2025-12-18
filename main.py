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
    if os.path.exists(savepath) and len(os.listdir(savepath)) != 2000:
        features_extract(
            audio_data=all_data, 
            savepath=savepath,
            frame_time=args.frame_time,
            hop_time=args.hop_time,
        )

    # 相似mfcc特征阵计算获取
    if len(os.listdir(savepath))==2000:
        respath = f"frame_time={args.frame_time}_hop_time={args.hop_time}_k={args.k}_sim={args.sim_mode}.txt"

        if not os.path.exists(f"./results/results_{args.test_mode}_top{args.k}"):
            os.makedirs(f"./results/results_{args.test_mode}_top{args.k}")
        if os.path.exists(os.path.join(f"./results/results_{args.test_mode}_top{args.k}", respath)):
            print("Eval data saved.")
            return True
        
        print(respath)
        accuracy = evaluate(
            test_data, 
            audio_data, 
            args.frame_time, 
            args.hop_time, 
            k=args.k, 
            sim_mode=args.sim_mode,
            test_mode=args.test_mode,
        )

        with open(os.path.join(f"./results/results_{args.test_mode}_top{args.k}", respath), "w") as resfile:
            resfile.write(f"Experiment Settings:\n")
            resfile.write(f"Frame Time: {args.frame_time}\n")
            resfile.write(f"Hop Time: {args.hop_time}\n")
            resfile.write(f"Similarity Mode: {args.sim_mode}\n")
            resfile.write(f"K: {args.k}\n")
            resfile.write("-" * 30 + "\n")
            resfile.write(f"Accuracy: {accuracy:.4f}\n")
        
        print(f"Results saved to {os.path.join(f'./results/results_{args.test_mode}_top{args.k}', respath)}")


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/ESC-50-master/audio")
    parser.add_argument("--test_mode", type=str, default="hit", choices=["hit", "score"])
    parser.add_argument("--frame_time", type=float, default=0.025)
    parser.add_argument("--hop_time", type=float, default=0.010)
    parser.add_argument("--sim_mode", type=str, default="DTW", help="sim mode", choices=["DTW", "cosine", "l2"])
    parser.add_argument("--k", type=int, default=5, help="k for knn")
    args = parser.parse_args()
    main(args)

