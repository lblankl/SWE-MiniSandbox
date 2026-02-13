
## Introduction

This document explains how to:

1.  Cache minisandbox environments for SWE-smith instances (following their image sharing pipeline).
2.  Validate the full SWE-smith dataset using the environments cache.
3.  Filter `passed` instances.
4.  Collect SFT trajectories and optionally balance them.

SWE-smith reuses images across instances to reduce storage. We follow the same pattern for environment preparation: venvs can be shared across instances with the same `image_name`.

___

## 1\. Identifying Unique Environments

First, extract instances with unique images from the SWE-smith dataset using the `image_name` field. This avoids redundant environment setup.

```bash
basedir=/home/zeta/SWE  # change to your own basedir
cd $basedir/SWE-MiniSandbox

python data/unique_images.py \
  --datap SWE-bench/SWE-smith-py \
  --save_path $basedir/SWE-MiniSandbox/dataset/SWE-smith-unique_images
```

> Note: The original dataset path we use in our paper is `SWE-bench/SWE-smith`, but it includes non-Python tasks that are not supported. Here we recommend using `SWE-smith-py` instead.

___

## 2\. Preparing Environment Cache for Unique Images

Each unique image will have a corresponding environment cache. The cache includes:

-   A **venv** (shared across instances with the same `image_name`)
-   An optional **git repo cache** (not necessary for SWE-smith; by default, we cache git repos here)

### Setup and Dependencies

```bash
basedir=/home/zeta/SWE  # change to your own basedir

unset PIP_CONSTRAINT
apt-get install -y graphviz

database=$basedir/SWE-MiniSandbox/dataset/SWE-smith-unique_images

conda_env=$basedir/miniconda3       # Conda with our minisandbox installed
env_dir=/home/zeta/SWE/conda        # Conda env used only for venv creation inside sandboxes
cache_dir=/home/smith               # Base directory for environment cache
output_dir=$cache_dir/out           # Directory for run results
sandbox_dir=$cache_dir/sandbox      # Root directory for sandbox environments
cached_git=$cache_dir/cached_git    # Git repo cache directory (optional)
shared_venv_dir=$cache_dir/shared_venv  # Directory for shared venvs

source $conda_env/bin/activate rl

pip install tomli tomli-w
pip install httpbin
```

### Run Environment Preparation

```bash
sweagent run-batch --config $basedir/SWE-MiniSandbox/config/swesmith_infer.yaml \
  --instances.type swesmith \
  --env_type sandbox \
  --instances.deployment.conda_env=$env_dir \
  --instances.deployment.delete_after_create=False \
  --agent.model.api_base http://0.0.0.0:8000/v1 \
  --random_delay_multiplier=1 \
  --instances.deployment.root_base=$sandbox_dir \
  --instances.deployment.tool_path=$basedir/SWE-MiniSandbox/SWE-agent/tools \
  --instances.deployment.git_base_path=$cached_git \
  --instances.deployment.shared_venv=$shared_venv_dir \
  --output_dir $output_dir \
  --instances.path $database \
  --instances.load_from_disk True \
  --instances.start 0 \
  --instances.end -1 \
  --instances.num_rollouts_per_instance -1 \
  --num_workers 60   # adjust parallelism based on your hardware
```

Only instances marked as `passed` in SWE-smith will successfully generate environment caches.

___

## 3\. Validating the Full SWE-smith Dataset

After caching environments for all unique images, you can validate the **full** SWE-smith dataset and reuse the cached environments.

The venv cache will automatically be reused for instances sharing the same `image_name`. Git repo caching is also available but not required for SWE-smith.

You can optionally pre-filter instances to only `passed` images, but for simplicity, this section uses the full dataset.

```bash
rm -rf $output_dir  # Clear previous output

database=SWE-bench/SWE-smith-py  # Full SWE-smith dataset
output_dir=$cache_dir/full_out   # Output directory for full run

sweagent run-batch --config $basedir/SWE-MiniSandbox/config/swesmith_infer.yaml \
  --instances.type swesmith \
  --env_type sandbox \
  --instances.deployment.conda_env=$env_dir \
  --instances.deployment.delete_after_create=False \
  --agent.model.api_base http://0.0.0.0:8000/v1 \
  --random_delay_multiplier=1 \
  --instances.deployment.root_base=$sandbox_dir \
  --instances.deployment.tool_path=$basedir/SWE-MiniSandbox/SWE-agent/tools \
  --instances.deployment.git_base_path=$cached_git \
  --instances.deployment.shared_venv=$shared_venv_dir \
  --output_dir $output_dir \
  --instances.path $database \
  --instances.load_from_disk False \  # Load from Hugging Face dataset
  --instances.start 0 \
  --instances.end -1 \
  --instances.num_rollouts_per_instance -1 \
  --num_workers 60
```

Again, only instances marked `passed` in SWE-smith will generate environment caches successfully. Instances that `failed` or other may do so due to environment setup issues.

To debug failed instances:

-   The install commands are mapped in
    
    [`get_install_commands_wrapper`](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/sandbox_deployment.py#L59).

-   The test commands are mapped in
    
    [`get_test_commands_wrapper`](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/sandbox_deployment.py#L66).

You can inspect logs and fix the environments manually or via an LLM-based assistant.

On subsequent runs of the same instance, the prepared environment cache will be reused.

___

## 3.5\. Rebuild Some Data Items (Optional)

Some instances may raise exceptions during environment setup (e.g., due to transient network issues, venv creation errors, Git clone failures, or package problems). In such cases, you can selectively rebuild only the failed instances (see [Rebuilding Failed Instances](swe-bench.md#rebuilding-failed-instances)). This is necessary when validating the RL environment cache in our image. However, you do not need to care about this for the RL process, as we have implemented automatic rebuild logic.

___

## 4\. Filtering `passed` Instances

Once the full dataset run is complete, you can filter out `passed` instances for training.

```bash
output_dir=/home/smith  # Directory containing run_batch_exit_statuses.yaml

cd $basedir/SWE-MiniSandbox

python data/filter_smith.py \
  --res_dir $output_dir \
  --dataset_path SWE-bench/SWE-smith-py \
  --output_path $basedir/SWE-MiniSandbox/dataset/smith-passed
```

You may delete git caches for failed instances to save storage.

___

## 5\. SFT Trajectory Collection

With `passed` instances filtered, you can now collect SFT trajectories as golden data for training.

### 5.1. Model API Deployment

Serve an LLM via API or use an existing endpoint. Update `agent.model.api_base` and `agent.model.name` accordingly in your config or commands.

Example: Serving `SWE-bench/SWE-agent-LM-32B` with vLLM:

```
bashmodelp=SWE-bench/SWE-agent-LM-32B
conda activate vllm  # create/use a vLLM environment

vllm serve $modelp \
  --tensor-parallel-size 8 \
  --async-scheduling \
  --served-model-name custom
```

This exposes an API at `http://0.0.0.0:8000/v1` with model name `custom`.

### 5.2. Golden Trajectory Collection

```bash
basedir=/home/zeta/SWE  # change to your own basedir

unset PIP_CONSTRAINT
database=$basedir/SWE-MiniSandbox/dataset/smith-passed

pip install flask

conda_env=$basedir/miniconda3
env_dir=/home/zeta/SWE/conda/
base_dir=/home/zeta/SWE
output_dir=$base_dir/out
rm -rf $output_dir  # remove the old output dir if it exists
sandbox_dir=$base_dir/sandbox
cached_git=$base_dir/cached_git
shared_venv_dir=$base_dir/shared_venv

source $conda_env/bin/activate
pip install tomli tomli-w
pip install httpbin

sweagent run-batch --config $basedir/SWE-MiniSandbox/config/swesmith_infer_default.yaml \
  --env_type sandbox \
  --instances.deployment.type sandbox \
  --instances.deployment.conda_env=$env_dir \
  --instances.deployment.delete_after_create False \
  --instances.deployment.tool_path $basedir/SWE-MiniSandbox/SWE-agent/tools \
  --agent.type sandbox \
  --agent.model.api_base http://0.0.0.0:8000/v1 \
  --agent.model.temperature 0.8 \
  --agent.step_limit 100 \   # Step limit to avoid infinite loops
  --random_delay_multiplier=1 \
  --instances.deployment.root_base=$sandbox_dir \
  --instances.deployment.git_base_path=$cached_git \
  --instances.deployment.shared_venv=$shared_venv_dir \
  --output_dir $output_dir \
  --instances.path $database \
  --instances.start 0 \
  --instances.end -1 \
  --instances.load_from_disk True \
  --num_workers 60 \
  --instances.num_rollouts_per_instance 4  # Collect 4 trajectories per instance
```

After the run, aggregate trajectories:

```bash
python -m swesmith.train.traj_mgr.collect_trajs \
  --traj_dir $output_dir \
  --out_path $basedir/SWE-MiniSandbox/dataset/smith-sft-trajs/dataset.jsonl
```

Only trajectories for `submitted` instances with reward `1` are kept as golden trajectories.

___

## 6\. (Optional) SFT Data Balancing

The collected trajectories are often imbalanced across instance IDs: easy instances tend to have more trajectories than hard ones. You can balance the SFT dataset using the provided script.

```bash
output_yaml=$output_dir/run_batch_exit_statuses.yaml 
json_path=$basedir/SWE-MiniSandbox/dataset/smith-sft-trajs/dataset.jsonl

python /home/zeta/SWE/SWE/data/balance_data.py \
  --json_path $json_path \
  --yaml_dir $output_yaml \
  --output_path $basedir/SWE-MiniSandbox/dataset/smith-sft-trajs/balanced_sft_data.jsonl
```

The final balanced SFT dataset is:

```text
$basedir/SWE-MiniSandbox/dataset/smith-sft-trajs/balanced_sft_data.jsonl
```

