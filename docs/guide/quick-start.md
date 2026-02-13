## Installation

## Requirements

### Environment Requirements

-   Python: Version >= 3.11
-   CUDA: Version >= 12.8

### System Level Requirements

-   Linux OS (Ubuntu 20.04+ recommended)
-   Namespace isolation support (`unshare --mount` command)
-   Bind mount support (`mount --bind` command)

## Install from Docker Image

We provide a pre-built Docker image that can be run directly:

```bash
docker pull lblankl/swe-minisandbox:latest
docker run -it --privileged lblankl/swe-minisandbox:latest /bin/bash  # --privileged is necessary
```

All environments and dependencies are pre-installed in the Docker image under `/home/zeta/SWE`.

We also provide the venv and git project cache used in our paper:

-   `/home/smith`: 1600 data items for RL
-   `/home/swebench`: 500 data items for SWE-bench evaluation

## Install from Source

Alternatively, you can install the package from source.

### Clone and Install the Repository

Because the speed performance of SWE-MiniSandbox relies on storage performance, we recommend using a high-performance SSD disk for storing the data and environment. Here we use `/home/zeta/SWE` as an example path.

```bash
# Set up installation directory (e.g., /home/zeta/SWE)
basedir=/home/zeta/SWE
mkdir -p $basedir
cd $basedir

# Clone the repository
git clone https://github.com/lblankl/SWE-MiniSandbox.git

# Create and activate conda environment
conda_env=/home/SWE/miniconda3 # fill in your conda path
source /home/SWE/miniconda3/bin/activate
conda create -n rl python=3.11 -y
conda activate rl

# Install the SWE-MiniSandbox packages
bash $basedir/SWE-MiniSandbox/sh/install.sh --basedir $basedir \
--conda_env $conda_env
```

### Set Up Conda Backends

Create a series of conda environments for different Python versions (used for venv creation). We'll prepare them under `/home/zeta/SWE/conda/`.

```bash
# Download environment files from HuggingFace
# Files will be downloaded under $basedir/conda_environment_files
hf download lblankl/MiniSandbox --local-dir $basedir/conda_environment_files

conda_sh_dir=$basedir/conda_environment_files/environment_files
# Set the target directory for conda environments (e.g., /home/zeta/SWE/conda)
tg_dir=/home/zeta/SWE/conda

# Create conda environments for Python 3.5 to 3.12
for version in 3.5 3.6 3.7 3.8 3.9 3.10 3.11 3.12; do
  sh_file=$conda_sh_dir/$version.sh
  tg_path=$tg_dir/$version/miniconda3
  bash $sh_file -b -p $tg_path
done

# (Optional)Update 3.11 and 3.9 conda environments with necessary packages
# This is required for SWE-Bench environment cache in our image but may be not necessary for yours
# Note: We use bash to avoid modifying the $PATH of the current shell. This is important if you want to run the rest scripts under the current shell.
bash $basedir/SWE-MiniSandbox/sh/conda_env.sh --tg-dir $tg_dir
```

### Final Directory Structure

After installation, your directory structure should look like this:

```yaml
/home/zeta/SWE/:
  - conda_environment_files/   # Conda environment files from HuggingFace
    - environment_files/
      - 3.5.sh
      - 3.6.sh
      - ...
      - 3.12.sh
  - conda/                     # Conda environments for different Python versions
    - 3.5/miniconda3
    - 3.6/miniconda3
    - ...
    - 3.12/miniconda3
  - SWE-MiniSandbox/          # The cloned repository
```

## Collect Golden Trajectories for SFT (Optional)

Teaching the model to follow the SWE-agent tool format is crucial for RL. You have three options:

### Option 1: Use Pre-trained Models

Use existing models from SWE-smith:

-   [SWE-bench/SWE-agent-LM-7B](https://huggingface.co/SWE-bench/SWE-agent-LM-7B)
-   [SWE-bench/SWE-agent-LM-32B](https://huggingface.co/SWE-bench/SWE-agent-LM-32B)

### Option 2: Use Existing Trajectories

Download the 6k SFT data used in our paper from HuggingFace:

```
bashhf download lblankl/swe-minisandbox-sft-data6k --local-dir $basedir/datasets/swe-minisandbox-sft-data6k
```

The data will be downloaded to `$basedir/datasets/swe-minisandbox-sft-data6k`. This data was collected from SWE-Agent-LM-32B.

> **Note:** Our experiments generated 6k golden SFT data from the SWE-smith dataset with SWE-Agent-LM-32B, which has modest data quality. For better data quality, consider using trajectories from [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), which were collected from claude-3-7-sonnet-20250219.

<!-- `SWE-bench/SWE-agent-LM-7B` can achieve 15% resolve rate on SWE-bench Verified. -->

### Option 3: Collect Your Own Data

If you want to collect data yourself or use other models, refer to the [Smith Data](data/swe-smith.md) guide.

## Supervised Fine-Tuning (SFT)

Here's a sample script to perform SFT with the collected data, using Qwen-3B-Coder-Instruct as an example:

```bash
export WANDB_API_KEY=your_wandb_api_key

config_path=$basedir/SWE-MiniSandbox/config/tune/swe_Qwen3bsandbox.yaml
tune run --nnodes 1 --nproc_per_node 8 full_finetune_distributed --config ${config_path}
```

> **Note:** Please modify the config file path according to your own installation path.

To use the SWE-bench/SWE-smith-trajectories trajectories, modify `$basedir/SWE-MiniSandbox/config/tune/swe_Qwen3bsandbox.yaml`:

```yaml
dataset:
  source: SWE-bench/SWE-smith-trajectories
  split: xml  # Available splits: tool, xml, ticks
```

## On-policy Reinforcement Learning (RL)

### Prerequisites
You can download the RL data used in our paper by:
```bash
bashhf download lblankl/swe-minisandbox-rl_formatted-1600 --local-dir $basedir/datasets/swe-minisandbox-rl_formatted-1600
```

Before running RL training, ensure you have:

1.  Collected the RL data and environment cache as described in the [Environment Cache OverView](data/overview.md) and [Smith Data](data/swe-smith.md) guide, **OR**
2.  Used the Docker image provided by us, which contains:
    -   RL data used in our paper: `/home/zeta/SWE/datasets/rl_formatted-1600`
    -   Environment cache: `/home/smith`

> **Important:** If using the Docker image's environment cache, follow the instructions in the [Smith Data](data/swe-smith.md) guide to verify all environments are working correctly.

### Reinforcement Learning with SWE-MiniSandbox
If you want to generate the RL formatted data yourself:

```bash
python /home/zeta/SWE/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/preprocess_swegym.py \
    --output_dir datasets/rl_formatted \  # target directory for the formatted data
    --source_dir data/smith-image_passed \  # the train dataset path (hf format)
    --eval_data_dir /us3/yuandl/dataset/SWE-bench/SWE-bench_Verified/data \ # the hf eval dataset path
    --load_from_disk --data_range 0 1600
```
Note that the --source_dir must be validated with the minisandbox environment to ensure all environments can be launched successfully.

#### One Node SWE RL

To run RL training with SWE-MiniSandbox:

```bash
# Step 1: Launch the Ray cluster with sufficient resources
# --tar-io   40 * 50 MB = 2 GB tar io bandwidth
# --sandbox  128 sandbox instances in parallel (16bcz * 8 rollout n)
bash /home/zeta/SWE/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/head.sh \
  --tar-io 40 \
  --sandbox 128 \
  --container 128 \
  --container-start-up 128 \
  --sandbox-start-up 128 \
  --env-path /home/zeta/SWE/miniconda3/bin/activate
# Step 2: Run the RL training script
bash $basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/run_swe_3B_sandbox.sh
```

> **Note:** Please modify the paths in the scripts according to your installation path. You also need to modify the `$basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/smith.yaml` config file. For detailed explanation of YAML config options, refer to the [SkyRL CLI Explanation](cli/skyrl.md) guide. 

#### Multi-Node Distributed RL

First launch the Ray cluster across multiple nodes with the following command (run the first command on the head node and the second command on each worker node)

Head:
```bash
bash /home/zeta/SWE/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/head.sh \
  --tar-io 40 \
  --sandbox 128 \
  --container 128 \
  --container-start-up 128 \
  --sandbox-start-up 128 \
  --env-path /home/zeta/SWE/miniconda3/bin/activate
```
Worker:
```bash
bash /home/zeta/SWE/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/worker.sh \
  --tar-io 40 \
  --sandbox 128 \
  --container 128 \
  --container-start-up 128 \
  --sandbox-start-up 128 \
  --env-path /home/zeta/SWE/miniconda3/bin/activate \
  --master-ip your_ray_head_ip \
  --master-port 6379
```

Then run the RL training script on the head node:

```bash
bash $basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/2node-3b-16bcz-32n.sh
```

Our framework distributes the minisandbox environment instances across multiple nodes and collects the results back to the head node for RL training. We recommend **128 minisandbox** per node (best in our experiments) and you can scale up to **N x 128 minisandbox** with N nodes.

### Reinforcement Learning with Container Deployment

#### Prerequisites

-   **Container server**: A server that supports Docker with internet access to pull necessary images
-   **Docker daemon access**: Usually GPU containers don't have access to the docker daemon for security reasons. You need to:
    1.  Set up a separate Docker server that exposes the docker daemon port (default: `2375`) to the network
    2.  Ensure your training machine can access this port

> **Hardware Note:** The server machine used in our experiments is a single-node 32-CPU machine with 2TB SSD. For large-scale, faster environment startup with containers (over 512 containers), a multi-node Kubernetes cluster is needed. For more information about RL training with Kubernetes, refer to [DeepSWE](https://www.together.ai/blog/deepswe).

#### Running RL Training with Containers

```bash
# Step 1: Launch the Ray cluster with sufficient resources
# --tar-io not used for container deployment, but keep it for compatibility
# --container 128 container instances in parallel (16bcz * 8 rollout n)
bash /home/zeta/SWE/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/host.sh \
  --tar-io 40 \
  --sandbox 128 \
  --container 128 \
  --container-start-up 128 \
  --sandbox-start-up 128 \
  --env-path /home/zeta/SWE/miniconda3/bin/activate
# Step 2: Run the RL training script
bash $basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/run_swe_3B_container.sh
```

> **Note:** You also need to modify the `$basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/smith_docker.yaml` config file. For detailed explanation of YAML config options, refer to the [SkyRL CLI Explanation](cli/skyrl.md) and [SWE-agent CLI Explanation](cli/sweagent.md) guide.

## Evaluation with SWE-Agent and SWE-Bench

### Inference with SWE-Agent

You can run the standard evaluation pipeline as follows:

#### Step 1: Serve the Model

First, serve the model you want to evaluate:

```bash
vllm serve $modelp --tensor-parallel-size 8 \
  --async-scheduling \
  --served-model-name custom
```

#### Configure Docker Settings

You need to specify the following arguments in `$basedir/SWE-MiniSandbox/config/sweagent_docker.yaml`:

```yaml
instances:
  deployment:
    type: docker
    host: your_docker_server_ip  # IP address of the Docker server
    # host: 127.0.0.1             # Use this for local Docker
    docker_args:
      - "--env"
      - "YOUR_ENVIRONMENT_SETUP_COMMANDS"  # Commands to run before starting the container
```

#### Step 2: Run Inference

Then run the inference script:

```bash
# Configuration
database=SWE-bench/SWE-bench_Verified
basedir=/home/zeta/SWE
output_dir=/us3/yuandl/SWE-bench/out-sweagent-7B

# Export the DOCKER_HOST environment variable to point to your remote Docker server
export DOCKER_HOST=your_docker_host

# Set up environment
unset PIP_CONSTRAINT
apt install -y graphviz
conda_env=$basedir/miniconda3
env_dir=/home/zeta/SWE/conda/
rm -rf $output_dir
source $conda_env/bin/activate rl

# Run SWE-agent batch inference
sweagent run-batch --config $basedir/SWE-MiniSandbox/config/sweagent_docker.yaml \
  --env_type container \
  --instances.repo_type pre \
  --instances.deployment.type docker \
  --agent.model.api_base http://0.0.0.0:8000/v1 \
  --agent.model.timeout 100 \
  --agent.step_limit 400 \
  --agent.reward False \
  --agent.total_time_limit 1200 \
  --output_dir $output_dir \
  --instances.database $database \
  --instances.start 0 \
  --instances.end 500 \
  --instances.deployment.eval_timeout 300 \
  --num_workers 10
```



### Evaluation with SWE-Bench

With the `preds.json` file generated from the inference step above, you can evaluate the performance:

```bash
export DOCKER_HOST=your_docker_host

python -m swebench.harness.run_evaluation \
  --predictions_path /path/to/your/preds.json \
  --max_workers 10 \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --report_dir /us3/yuandl/SWE-bench/out/out-qwen2.5-3Bcoder-docker6k \
  --run_id out-qwen2.5-3Bcoder-docker6k
```

> **Note:** Replace `/path/to/your/preds.json` with the actual path to your predictions file.

<!-- ___

## Additional Resources

-   [Environment Preparation Guide](data/overview.md)
-   [MiniSandbox CLI API Guides](cli/sweagent.md) -->
