# test frame_time = [0.040, 0.025, 0.015]
# test   hop_time = [0.020, 0.010, 0.005]

CUDA_VISIABLE_DEVICES=0 python main.py \
    --test_mode hit \
    --frame_time 0.015 \
    --hop_time 0.005 \
    --k 10 \
    --sim_mode DTW
