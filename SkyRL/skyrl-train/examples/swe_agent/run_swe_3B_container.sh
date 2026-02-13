set -x
basedir=/home/zeta/SWE
source $basedir/miniconda3/bin/activate
conda activate rl
cd $basedir/SWE-MiniSandbox/SkyRL/skyrl-train

name=Qwen2.5-7B-coder-docker-5ksft-rl1600

export WANDB_API_KEY=""

unset PIP_CONSTRAINT
docker_host_ip=
export DOCKER_HOST=tcp://$docker_host_ip:2375

export no_proxy="localhost,192.168.171.254,$docker_host_ip,127.0.0.1,0.0.0.0,registry-1.docker.io,::1"
export NO_PROXY="localhost,192.168.171.254,$docker_host_ip,127.0.0.1,0.0.0.0,registry-1.docker.io,::1"
DATA_DIR="/home/zeta/SWE/datasets/rl_formatted-1600"
CKPT_PATH="/us3/yuandl/ckpt/$name"
Model_PATH="/us3/yuandl/ckpt/qwen2p5-coder-3b-containersft5k/epoch_1"
# Save trajectories here for debugging
# NOTE: For a multi-node cluster, ensure that this is on NFS so that you can save all trajectories in the same path


NUM_GPUS=8
NNODES=1
NUM_INFERENCE_ENGINES=4
TP_SIZE=2
LOGGER=wandb

env_dir=/home/zeta/SWE/conda/

output_dir=/upfs/yuandl/SWE/SkyRL-data/$name

# We use a small batch size here for demonstration
# NOTE (sumanthrh): The `generator.max_turns` here is actually unused, and we use the `step_limit` from the `swebench.yaml` file. 
# This simply has to be a value > 1
#uv run --isolated --extra vllm --extra miniswe --env-file examples/mini_swe_agent/.env.miniswe -m examples.mini_swe_agent.main_mini_swe \
python -m examples.swe_agent.main_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$Model_PATH \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=2 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=1 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=30 \
  trainer.max_prompt_length=16384 \
  trainer.asynch=False \
  generator.sampling_params.max_generate_length=4096  \
  generator.max_input_length=16384 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host='127.0.0.1' \
  generator.http_endpoint_port=8001 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="swerl2" \
  trainer.run_name=$name \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  +generator.miniswe_config_path="examples/mini_swe_agent/swebench.yaml" \
  +generator.sweagent.config="$basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/smith_docker.yaml" \
  +generator.sweagent.agent.type="default" \
  +generator.sweagent.agent.step_limit=150 \
  +generator.sweagent.agent.total_time_limit=300 \
  +generator.sweagent.agent.model.api_base=http://0.0.0.0:8001/v1 \
  +generator.sweagent.agent.model.name=openai/$Model_PATH \
  +generator.sweagent.agent.model.max_input_tokens=16383 \
  +generator.sweagent.agent.model.timeout=100 \
  +generator.sweagent.random_delay_multiplier=1 \
  +generator.sweagent.instances.type=skyrl \
  +generator.sweagent.output_dir=$output_dir \
  +generator.sweagent.instances.deployment.type=docker \
  +generator.sweagent.instances.deployment.eval_timeout=300 \
  +generator.sweagent.instances.repo_type=pre
# docker stop $(docker ps -aq)

# docker rm -f $(docker ps -aq)
# docker container prune -y

