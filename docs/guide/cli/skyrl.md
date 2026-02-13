# SkyRL CLI Guide

This guide explains how to use the SkyRL command-line interface (CLI) to run SWE-Agent reinforcement learning experiments.  
We integrate SWE-Agent configuration into SkyRL so that you can manage your RL experiments using SWE-Agent configuration files.

## SWE-Agent Configuration in SkyRL

SkyRL uses Hydra for configuration management.  
SWE-Agent–related parameters are specified under the `generator.sweagent` section in the SkyRL configuration. This lets you define all required SWE-Agent settings directly in the SkyRL config file.

For example:
```bash
python -m examples.swe_agent.main_swe \
    ...
  trainer.asynch=False \    # Our asynch implementation, still under development, must be set to False for now
  +generator.sweagent.config="$basedir/SWE/SkyRL/skyrl-train/examples/swe_agent/smith.yaml" \ # the base YAML config for SWE-Agent, which sets default values for all required parameters.
  +generator.sweagent.agent.type="skysbdefault" \  # must be set to skysbdefault for SkyRL SWE-agent
  +generator.sweagent.agent.step_limit=150 \  # step limit for each trajectory
  +generator.sweagent.agent.total_time_limit=300 \ # total time limit for each trajectory
  +generator.sweagent.agent.pre_check=True \  # whether to check the environment before running the agent. False for faster training.
  +generator.sweagent.agent.model.api_base=http://0.0.0.0:8001/v1 \
  +generator.sweagent.agent.model.name=openai/$Model_PATH \
  +generator.sweagent.agent.model.max_input_tokens=16384 \
  +generator.sweagent.agent.model.timeout=100 \ # the timeout for each model call
  +generator.sweagent.instances.deployment.eval_timeout=300 \ # If you want to calculate the reward, this defines the timeout for reward calculation in seconds.
  +generator.sweagent.instances.deployment.root_base=$sandbox_dir \
  +generator.sweagent.instances.deployment.git_base_path=$cached_git \
  +generator.sweagent.instances.deployment.shared_venv=$shared_venv_dir \
  +generator.sweagent.instances.deployment.tool_path=$basedir/SWE/SWE-agent/tools \
  +generator.sweagent.instances.type=skyrl \ # dataset type must be set to skyrl for SkyRL experiments
  +generator.sweagent.output_dir=$output_dir \
  +generator.sweagent.instances.deployment.conda_env=$env_dir \
  +generator.sweagent.instances.deployment.type=sandbox \ # deployment type must be set to sandbox for minisandbox
  +generator.sweagent.instances.repo_type=github \ # repo type must be set to github for minisandbox
  +generator.sweagent.env_type=sandbox \ # environment type must be set to sandbox for minisandbox
  ...
```