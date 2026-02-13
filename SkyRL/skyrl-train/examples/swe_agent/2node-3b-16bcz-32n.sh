set -x
basedir=/home/zeta/SWE
source $basedir/miniconda3/bin/activate rl
cd $basedir/SWE-MiniSandbox/SkyRL/skyrl-train

export WANDB_API_KEY=

unset PIP_CONSTRAINT

bcz=16
rollout_n=16
name=2node${bcz}bcz-${rollout_n}

DATA_DIR=
CKPT_PATH=
Model_PATH=

NUM_GPUS=8
NNODES=2
NUM_INFERENCE_ENGINES=4
TP_SIZE=4
LOGGER=wandb

env_dir=/home/zeta/SWE/conda/
cache_dir=/home/smith
sandbox_dir=$cache_dir/$name
rm -rf sandbox_dir
cached_git=$cache_dir/cached_git
shared_venv_dir=$cache_dir/shared_venv
THIS_IP=$(hostname -I | awk '{print $1}')
output_dir=
# rm -rf $output_dir



python -m examples.swe_agent.main_swe \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$Model_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  trainer.policy.sequence_parallel_size=4 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=1 \
  trainer.eval_batch_size=50 \
  trainer.eval_before_train=true \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$bcz \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=30 \
  trainer.max_prompt_length=16384 \
  generator.sampling_params.max_generate_length=4096  \
  generator.max_input_length=16384 \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=True \
  generator.enable_http_endpoint=True \
  generator.http_endpoint_host=$THIS_IP \
  generator.http_endpoint_port=8001 \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=$rollout_n \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="minisandbox" \
  trainer.run_name=$name \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.asynch=False \
  +generator.sweagent.config="$basedir/SWE-MiniSandbox/SkyRL/skyrl-train/examples/swe_agent/smith.yaml" \
  +generator.sweagent.agent.type="skysbdefault" \
  +generator.sweagent.agent.step_limit=150 \
  +generator.sweagent.agent.total_time_limit=300 \
  +generator.sweagent.agent.pre_check=False \
  +generator.sweagent.agent.model.api_base=http://$THIS_IP:8001/v1 \
  +generator.sweagent.agent.model.name=openai/$Model_PATH \
  +generator.sweagent.agent.model.max_input_tokens=16384 \
  +generator.sweagent.agent.model.timeout=100 \
  +generator.sweagent.instances.deployment.eval_timeout=300 \
  +generator.sweagent.instances.deployment.root_base=$sandbox_dir \
  +generator.sweagent.instances.deployment.git_base_path=$cached_git \
  +generator.sweagent.instances.deployment.shared_venv=$shared_venv_dir \
  +generator.sweagent.instances.deployment.tool_path=$basedir/SWE-MiniSandbox/SWE-agent/tools \
  +generator.sweagent.instances.type=skyrl \
  +generator.sweagent.output_dir=$output_dir \
  +generator.sweagent.instances.deployment.conda_env=$env_dir \
  +generator.sweagent.instances.deployment.type=sandbox \
  +generator.sweagent.instances.repo_type=github \
  +generator.sweagent.env_type=sandbox
