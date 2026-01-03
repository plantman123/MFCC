import argparse
import os
from function import (
    load_esc50,
    features_extract,
    evaluate,
)


def main(args):
    savepath = os.path.join("./features", f"frame_time={args.frame_time}_hop_time={args.hop_time}_n_mels={args.n_mels}_n_ceps={args.n_ceps}_lifter={args.lifter}_norm={args.norm}")
    audio_data, test_data, all_data = load_esc50()

    # 特征提取存储(如果还没有保存)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        features_extract(
            audio_data=all_data, 
            savepath=savepath,
            frame_time=args.frame_time,
            hop_time=args.hop_time,
            n_mels=args.n_mels,
            n_ceps=args.n_ceps,
            lifter=args.lifter,
            norm=args.norm,
        )
    if os.path.exists(savepath) and len(os.listdir(savepath)) != 2000:
        features_extract(
            audio_data=all_data, 
            savepath=savepath,
            frame_time=args.frame_time,
            hop_time=args.hop_time,
            n_mels=args.n_mels,
            n_ceps=args.n_ceps,
            lifter=args.lifter,
            norm=args.norm,
        )

    # 相似mfcc特征阵计算获取
    if len(os.listdir(savepath))==2000:
        respath = f"frame_time={args.frame_time}_hop_time={args.hop_time}_n_mels={args.n_mels}_n_ceps={args.n_ceps}_lifter={args.lifter}_norm={args.norm}_k={args.k}_sim={args.sim_mode}.txt"
        
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
            dist_mode=args.sim_mode,
            test_mode=args.test_mode,
            n_mels=args.n_mels,
            n_ceps=args.n_ceps,
            lifter=args.lifter,
            norm=args.norm,
        )

        with open(os.path.join(f"./results/results_{args.test_mode}_top{args.k}", respath), "w") as resfile:
            resfile.write(f"Experiment Settings:\n")
            resfile.write(f"Frame Time: {args.frame_time}\n")
            resfile.write(f"Hop Time: {args.hop_time}\n")
            resfile.write(f"N Mels: {args.n_mels}\n")
            resfile.write(f"N Ceps: {args.n_ceps}\n")
            resfile.write(f"Lifter: {args.lifter}\n")
            resfile.write(f"Norm: {args.norm}\n")
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
    parser.add_argument("--lifter", type=int, default=22, help="lifter coefficient")
    parser.add_argument("--norm", type=str, default="cms", choices=["cmvn", "cms", "none"], help="normalization method")
    parser.add_argument("--sim_mode", type=str, default="DTW", help="sim mode", choices=["DTW", "cosine", "l2"])
    parser.add_argument("--k", type=int, default=5, help="k for knn")
    parser.add_argument("--n_mels", type=int, default=40, help="number of mel filters")
    parser.add_argument("--n_ceps", type=int, default=20, help="number of mfcc coefficients")
    args = parser.parse_args()
    main(args)

