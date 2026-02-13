# SWE-smith Training Code
This folder contains the training scripts for fine-tuning on SWE-smith trajectories.

The code is heavily inspired by the [SWE-gym](https://github.com/SWE-Gym/SWE-Gym) team. We thank them for open-sourcing their codebase, allowing for easy reproduction of the fine-tuning procedure they used.
If you found this part of the codebase useful, please make sure to [cite the SWE-gym](https://github.com/SWE-Gym/SWE-Gym?tab=readme-ov-file#-citation) team as well.

### Notes
All fine-tuning + model serving is carried out with [Modal](https://modal.com/).

To fine tune a model, follow this procedure:
1. Download a model checkpoint from HuggingFace
```bash
modal run download_checkpoint.py --source-repo Qwen/Qwen2.5-7B-Instruct --target-dir /weights/Qwen/Qwen2.5-7B-Instruct
```

2. Run fine tuning with a SWE-smith dataset
```bash
NGPUS=8 modal run train/run_ft_torchtune.py --config train/config/torchtune.yml
```

3. Host model with `sglang` and run inference with SWE-agent
```bash
N_HOURS=4 N_GPUS=4 modal run --detach serve_sglang.py --model-path /weights/my-oss-model --served-model-name gpt-4o --tokenizer-path /weights/Qwen/Qwen2.5-Coder-32B-Instruct
```

From the SWE-agent local repository, run the following command to start running inference with the local model
```bash
#!/bin/bash

sweagent run-batch \
    --agent.model.api_base <REPLACE WITH MODAL LINK>/v1 \
    --agent.model.api_key swesmith \ # This is set in serve_sglang.py
    --agent.model.name gpt-4o \        # TODO: Change this when SWE-agent is fixed
    --instances.type swe_bench \
    --instances.dataset_name jyang20/swebv-mini \
    --instances.split test \
    --config config/anthropic_no_fcalls.yaml
    # --instances.evaluate True # Install sb-cli for this
```
