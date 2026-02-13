
# Introduction

This section introduces the environment pre-cache process for **SWE-MiniSandbox**.  
The environment cache is composed of two main parts:

1. Venv-based Python environment cache  
2. GitHub repo cache (optional)

---

## Venv-Based Python Environment Cache

We pre-install a list of `miniconda3` environments with different Python versions (see [Conda Backends](../quick-start.md#set-up-conda-backends)). These conda environments are used as bases to create virtual environments (venvs) for specific tasks.

When creating a new sandbox, SWE-MiniSandbox:

1. Selects a conda environment according to the task’s Python version requirement.
2. Creates a new venv based on that conda environment and install the dependencies.
3. Packs the venv and stores it under a shared directory.

The venv directory structure is:

```yaml
shared_venv_dir/
  repo_id_1/               # The repo id of the task
    python_version/        # The python version
      image_name_1/        # The image name of the task
        venv.tar.gz        # Packed venv directory
      image_name_2/
        venv.tar.gz
  repo_id_2/
    python_version/
      image_name_1/
        venv.tar.gz
```

So the path to the venv of a specific task is:

```text
{shared_venv_dir}/{repo_id}/{python_version}/{image_name}/venv.tar.gz
```

`shared_venv_dir` is user-defined. **Do not change this directory name after pre-cache**, because the path is currently hard-coded inside the venv. (A more flexible mechanism will be implemented in the future.)

### How `repo_id` Is Determined
Every deployment config gets a data item `ds` from the dataset.


The `repo_id` is obtained through the mapping function `swesandbox.sandbox_deployment.map_to_git_id`:

```python
def map_to_git_id(ds, data_type):
    if data_type == "swebench":
        return ds["repo"]
    elif data_type == "swesmith":
        if "repo" in ds:
            return ds["repo"]
        instance_id = ds["instance_id"]
        git_id = ".".join(instance_id.split(".")[:2])
        return git_id
    else:
        return ds["repo"]
```

### How `python_version` Is Determined

The `python_version` is obtained from the dataset:

```python
python_version = self._config.ds.get("version", "latest")
```

If not specified, it defaults to `"latest"`.

### How `image_name` Is Determined

The `image_name` is obtained from the dataset:

```python
image_name = self._config.ds.get("image_name", "default")
```

If not specified, it defaults to `"default"`.

___

## GitHub Repo Cache

After installing the GitHub repo into the corresponding venv, we can cache the repository on local disk. This is:

-   **Necessary** for the SWE-bench dataset
-   **Optional** (not necessary) for SWE-smith

The GitHub repo cache directory structure is:

```yaml
cached_git/
  repo_id_1/               # The repo id of the task
    python_version/        # The python version
      instance_id_1/       # The instance id of the task
        testbed.tar.gz     # Packed GitHub repo
      instance_id_2/
        testbed.tar.gz
  repo_id_2/
    python_version/
      instance_id_1/
        testbed.tar.gz
```

The path to the cached GitHub repo for a specific task is:

```text
{cached_git_dir}/{repo_id}/{python_version}/{instance_id}/testbed.tar.gz
```

The `instance_id` is obtained from the dataset:

```python
instance_id = ds.get("instance_id", "default")
```

### Disabling GitHub Repo Cache

If you do not want to cache the GitHub repo, set the `cache_git` parameter to `False` in the deployment config class [`SandboxDeploymentConfig`](https://idealab.alibaba-inc.com/api/swe-minisandbox.md#swesandbox.sandbox_deployment.SandboxDeploymentConfig).

-   If `cache_git = False`, the GitHub repo is fetched directly from the remote repository and installed in editable mode into the venv.
-   If `cache_git = True`, the fetched and installed repo is packed and stored as described above.

---

For SWE-Bench, `cache_git` needs to be set to `True` to ensure correct functionality.
For SWE-Smith, it can be set to `False` to save storage space. By default, it is set to `True`.
___

## Environment Pre-Cache Pipeline

The environment pre-cache pipeline is implemented in the class [`sweagent.agent.empty_agent.EmptyAgent`](../api/sweagent/agent.md#sweagent.agent.empty_agent.EmptyAgent).

Workflow:

1.  Prepare the environment (venv + repo).
2.  Run an evaluation script to validate the environment.
3.  Mark each environment as `passed` or `failed` or other statuses.

Only environments marked as `passed` are used in subsequent training and evaluation.

For failed environments, you can inspect the error logs and prediction files (`output_dir/instance_id/exception.log` and `output_dir/instance_id/instance_id.pred`) to debug manually or with large language model assistance (an automatic pipeline will be provided in the future).

### Pre-Cache Status Outputs

The status of environment pre-cache is stored under the `output_dir`, with the following structure:

```yaml
output_dir/
  instance_id_1/
    instance_id_1.config.yaml  # Config file used to set up the environment
    instance_id_1.debug.log    # Debug log for environment setup (not used by default)
    instance_id_1.pred         # Prediction JSON from the evaluation script. Key fields:
                               #   - reward : 1 if passed, 0 if failed
                               #   - test_out : output of the evaluation script
                               #   - p2p : success rate of pass-to-pass cases
                               #   - f2p : success rate of fail-to-pass cases
    instance_id_1.traj         # Agent trajectory; empty for EmptyAgent
    (exception.log)            # If any exception occurs during setup, it is stored here
  instance_id_2/
    ...
  preds.json                   # Aggregated predictions of all instances (not useful for EmptyAgent)
  run_batch_exit_statuses.yaml # Exit status of all instances:
                               #   - "passed" / "failed" / other Exceptions for EmptyAgent
                               #   - "submitted" / other Exceptions for other agents
```