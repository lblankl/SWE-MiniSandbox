source /upfs/yuandl/SWE/miniconda3/bin/activate
conda activate rl
# python -m skyrl_train.model_merger merge \
#     --backend fsdp \
#     --tie-word-embedding \
#     --local_dir /us3/yuandl/ckpt/Qwen2.5-3B-coder-docker-5ksft-rl1600-300timeout/global_step_101/policy \
#     --target_dir /us3/yuandl/ckpt/Qwen2.5-3B-coder-docker-5ksft-rl1600-300timeout/global_step_101/models \
#     --hf_model_config_path /us3/yuandl/ckpt/qwen2p5-coder-3b-dockersft5k/epoch_1 \
#     --use_cpu_initialization

python -m skyrl_train.model_merger merge \
    --backend fsdp \
    --tie-word-embedding \
    --local_dir /us3/yuandl/ckpt/Qwen2.5-7B-coder-docker-5ksft-rl1600/global_step_101/policy \
    --target_dir /us3/yuandl/ckpt/Qwen2.5-7B-coder-docker-5ksft-rl1600/global_step_101/models \
    --hf_model_config_path /us3/yuandl/ckpt/qwen2p5-coder-7b-dockersft5k/epoch_1 \
    --use_cpu_initialization