N_GPUS=8 modal run --detach swesmith/train/run/ft_torchtune.py --config configs/train/full_ft_qwen_32b.yml

# N_GPUS=2 modal run --detach swesmith/train/run/ft_torchtune.py --config configs/train/full_ft_qwen_7b.yml