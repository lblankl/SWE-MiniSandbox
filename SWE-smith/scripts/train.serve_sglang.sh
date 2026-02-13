#!/bin/bash

# N_HOURS=4 N_GPUS=4 modal run --detach swesmith/train/serve_sglang.py \
#     --model-path /weights/outputs/qwen2p5-coder-32b-lora-lr1e-4-warmup5___difficulty/qwen2p5-coder-32b-lora-lr1e-4-warmup5___difficulty/merged \
#     --served-model-name gpt-4o \
#     --tokenizer-path /weights/Qwen/Qwen2.5-Coder-32B-Instruct

N_HOURS=3 N_GPUS=8 modal run --detach /ossfs/workspace/nas/agent/SWE-smith/swesmith/train/serve_sglang.py \
    --model-path /home/zeta/models/SWE-bench/SWE-agent-LM-32B \
    --served-model-name gpt-4o \
    --tokenizer-path /weights/Qwen/Qwen2.5-Coder-32B-Instruct
