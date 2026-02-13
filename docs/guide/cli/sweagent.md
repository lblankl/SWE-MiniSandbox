# SWE-Agent Cli Guides
This guide explains how to use the SWE-Agent API to run batch inference with the SWE-MiniSandbox framework.

SWE-Agent organizes its CLI commands using Pydantic models, which makes it easy to configure experiments via YAML files.  
We provide several example YAML configuration files in the [config directory](https://github.com/lblankl/SWE-MiniSandbox/tree/main/config) to help you get started.

To specify a YAML configuration file, use the `--config` flag when running the CLI command. For example:

```bash
sweagent run-batch --config $basedir/SWE-MiniSandbox/config/sweagent_docker.yaml
```

## Run Batch Configuration
Commonly used configuration parameters include:
```yaml
num_workers: 80   # number of parallel workers to run the batch inference
output_dir : /home/smith/out  # directory to save the log and output of the batch inference
env_type : sandbox  # type of the environment to use, can be 'sandbox' or 'docker'
```
You can also specify them directly in the command line:
```bash
sweagent run-batch --config $basedir/SWE-MiniSandbox/config/sweagent_docker.yaml \
    --num_workers 80 \
    --output_dir /home/smith/out \
    --env_type docker
```

## Instances Configuration

The `instances` field in the configuration specifies the dataset and deployment settings used for batch inference.

For example, in a SWE-MiniSandbox YAML configuration file:
```yaml
instances:
  type: swesmith # the type of the dataset, can be 'swesmith' , 'swe_bench' or 'skyrl'
  path: /path/to/dataset    # path to the smith format dataset, only needed when the dataset type is set to 'swesmith' and 'skyrl'
 
  load_from_disk: true   # whether to load the dataset from disk, set to false if you want to load the dataset using load_dataset function.
  split: train # the split of the dataset to use
  deployment:
    type: sandbox # the type of the deployment environment, 'sandbox' for MiniSandbox, 'docker' for docker container
    repo_type: github  #Type of the repository. Can be 'github', 'local', or anything else for pre-existing repositories. MiniSandbox currently only supports 'github'. 'local` is not supported yet. Container only supports 'pre-existing'. 
    root_base: /home/smith/sandbox  # the root base directory for the sandbox deployment, all the minisandbox instances will be created under this directory. Only needed when the deployment type is set to 'sandbox'.
    git_base_path: /home/smith/gitcache # the github cache directory for the sandbox deployment, all the git repositories will be cached under this directory. Only needed when the deployment type is set to 'sandbox'.
    data_type: swesmith  # the type of the data, can be 'swesmith' or 'swe_bench', 
    tool_path: /home/zeta/SWE/SWE-MiniSandbox/SWE-agent/tools  # the path to the tools directory. We use `tools` folder for MiniSandbox. MiniSandbox tool implementation is modified slightly.
    conda_env: /home/zeta/SWE/conda/  # The conda backend directory to be mounted to the deployment. Important for venv management.
    abs_tool_path: /tools # which folder do you want to copy the tool_path to in the MiniSandbox deployment.
    shared_venv: /home/smith/shared_venv # The venv cache directory for the sandbox deployment, all the venvs will be cached under this directory. 
    git_folder: testbed  # The folder name for the git repository in the deployment.
    eval_timeout: 300 # If you want to calculate the reward, this defines the timeout for reward calculation in seconds.
    
```
in a Container YAML configuration file:
```yaml
instances:
  type: swebench # the type of the dataset, can be 'swesmith' , 'swe_bench' or 'skyrl'
  database: /path/to/database  # path to the SWE-bench Verified dataset, only needed when the dataset type is set to 'swe_bench'
  split: test  # the split of the dataset to use
  deployment:
    type: docker   # set to docker for running the batch inference in docker containers
    host: 192.168.0.176  # the host ip for the container server
    eval_timeout: 300 # If you want to calculate the reward, this defines the timeout for reward calculation in seconds.
    docker_args:
      - "--env"
      - "echo $HTTP_PROXY"  # Additional commands to be passed to the `docker run` command when creating the container. For example, you can use this to set up proxy for the container.
```

You can also specify the instances configuration directly in the command line using dot notation. For example:
```bash
sweagent run-batch --config $basedir/SWE-MiniSandbox/config/sweagent_docker.yaml \
    --instances.type swesmith \
    --instances.path /path/to/dataset
```
**Note**: Some configuration parameters can only be set in the YAML file and not in the command line due to the complexity of the nested configuration structure. For example, `instances.deployment.type`, `instances.deployment.docker_args`.

## Agent Configuration

We implement multiple agent classes. The `agent.type` field in the configuration specifies which agent class to use for batch inference. Currently, the available types are:

- `default`: The default container-based agent for SWE agent tasks.
- `empty`: A no-op agent that does nothing. Useful for MiniSandbox environment caching and testing.
- `sandbox`: The default agent for MiniSandbox-based run.
- `skysbdefault`: The MiniSandbox agent for Sky-RL. It is similar to `sandbox` but includes minor modifications for Sky-RL workflows.

An example configuration:
```yaml
agent:
  type: sandbox  # the type of the agent, can be 'default', 'empty', 'sandbox', 'skysbdefault'
  step_limit: 150  # the maximum number of agent steps to run for each instance.
  total_time_limit: 300 # the maximum total time limit (in seconds) for each instance.
  templates:
    ...
    #templates can be defined here, we use default templates. Do not change.
  tools:
    bundles: # The tools to be included in the agent. `skysbdefault`, `empty`, `sandbox` must use tools/... The `default` agent must use tool/...
      - path: tools/registry  
      - path: tools/edit_anthropic
      - path: tools/submit
    env_variables:
      USE_FILEMAP: 'true'  # Do not change
    enable_bash_tool: true  # Do not change
    parse_function:
      type: xml_function_calling  # Do not change if you serve custom models yourself.
    str_replace_editor: # Do not change
      arguments:
      - name: view_range
        argument_format: "--view_range {{value}}"
    execution_timeout: 300  # The timeout for each tool execution in seconds.
  history_processors:
    - type: last_n_observations
      n: 5
  model:
    api_key: ssgsd  # The API key for the model. Not necessary if you serve custom models yourself.
    api_base: http:// # The API url for the model.
    name: custom_openai/custom  # The name of the model to use. Depends on your model serving setup.
    per_instance_cost_limit: 0
    per_instance_call_limit: 0
    temperature: 0.8  # The temperature for the model.
    max_input_tokens: 16383  # The maximum input tokens for the model.
    timeout: 100  # The timeout for each model call in seconds.
```

For more details on the sweagent configuration, please refer to the [SWE-AgentAPI Reference](../api/sweagent/swe-agent.md).