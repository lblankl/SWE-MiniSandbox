
## Introduction

SWE-bench Verified uses a one-to-one mapping between each instance and its environment image: every instance has its own unique environment. This makes environment cache preparation simpler than in SWE-smith.

## Validating the Full Dataset

You can run the full SWE-bench Verified dataset to generate the environment cache. A separate virtual environment and GitHub repository cache will be created for each instance.

Because this setup assumes a **restricted network environment** (where only whitelisted URLs are accessible on our GPU machines), some instances that require external network access must be handled specially:

-   A local HTTP proxy server (`httpbin`) is started to serve requests that would otherwise go to some public url.
-   Some required resources (e.g., qhull source code) are pre-downloaded and copied locally.
-   The original SWE-bench Verified environment installation commands are modified to be compatible with this restricted environment.

### Environment Cache Preparation

Use the following command to prepare the cache:

```bash
unset PIP_CONSTRAINT

database=/upfs/xxx/SWE/SWE/dataset/SWE-bench/SWE-bench_Verified/data   # path to your SWE-bench Verified dataset
basedir=/home/zeta/SWE                                                   # change to your own base dir

apt-get update
apt-get install -y inkscape
apt-get install -y graphviz
mkdir -p /root/.cache/pip/wheels

zip_to_copy=$basedir/SWE-MiniSandbox/zip/qhull-2020-src-8.0.2.tgz
if [ ! -f /tmp/qhull-2020-src-8.0.2.tgz ]; then
  cp "$zip_to_copy" /tmp/qhull-2020-src-8.0.2.tgz
fi

conda_env=$basedir/miniconda3
env_dir=/home/zeta/SWE/conda/
base_dir=/home/swebench
output_dir=/upfs/xxx/SWE/swebench_cache/out/swebench_cacheout
rm -rf "$output_dir"
sandbox_dir=$base_dir/sandbox
cached_git=$base_dir/cached_git
shared_venv_dir=$base_dir/shared_venv

source "$conda_env/bin/activate" rl
pip install tomli tomli-w
pip install httpbin

python -m httpbin.core --port=5000 &
HTTPBIN_PID=$!

sleep 2

# Ensure httpbin is killed when the script exits
trap "kill $HTTPBIN_PID" EXIT

sweagent run-batch --config "$basedir/SWE-MiniSandbox/config/sweagent_infer.yaml" \
    --env_type sandbox \
    --instances.deployment.conda_env="$env_dir" \
    --agent.model.api_base http://0.0.0.0:8000/v1 \
    --random_delay_multiplier=1 \
    --instances.deployment.root_base="$sandbox_dir" \
    --instances.deployment.tool_path="$basedir/SWE-MiniSandbox/SWE-agent/tools" \
    --instances.deployment.git_base_path="$cached_git" \
    --instances.deployment.shared_venv="$shared_venv_dir" \
    --output_dir "$output_dir" \
    --instances.database "$database" \
    --instances.start 0 \
    --instances.end 500 \
    --num_workers 10

kill $HTTPBIN_PID
```

The environment cache is considered successfully prepared only if **all instances are marked as `"passed"`** in the output logs.

### Rebuilding Failed Instances

Some instances may fail during environment setup (e.g., due to transient network issues, venv creation errors, Git clone failures, or package problems). In that case, you can **selectively rebuild** only the failed instances.

Assume `sphinx-doc__sphinx-7757` and `django__django-12125` failed in a previous run. You can rebuild them with:

```bash
# Same setup (env vars, httpbin, etc.) as above

sweagent run-batch --config "$basedir/SWE-MiniSandbox/config/sweagent_infer.yaml" \
    --env_type sandbox \
    --instances.deployment.conda_env="$env_dir" \
    --agent.model.api_base http://0.0.0.0:8000/v1 \
    --random_delay_multiplier=1 \
    --instances.deployment.root_base="$sandbox_dir" \
    --instances.deployment.tool_path="$basedir/SWE-MiniSandbox/SWE-agent/tools" \
    --instances.deployment.git_base_path="$cached_git" \
    --instances.deployment.shared_venv="$shared_venv_dir" \
    --instances.deployment.eval_timeout=500 \  # Some cases fails due to not enough evaluation time limit. The default is 300
    --output_dir "$output_dir" \
    --instances.database "$database" \
    --instances.deployment.force_rebuild=True \ # rebuild only specified instances specified below, if you want to reuse the cache, do not set it True
    --instances.filter "^(sphinx-doc__sphinx-7757|django__django-12125)$" 

kill $HTTPBIN_PID  # do not forget to kill the httpbin server
```
In our image, we have guaranteed that all instances in SWE-bench Verified can be successfully cached (We recommand checking it again by yourself. To reuse the cache, --instances.deployment.force_rebuild must be set to False). However, **not all SWE-bench Verified instances are guaranteed to succeed** in your own image. Some may still fail due to:

-   Network limitations
-   Incompatible or broken pip packages
-   Project-specific build or test issues

You can inspect the logs to debug these failures manually or with LLM assistance.

The mapping from an instance ID to its environment setup commands is defined in

[`get_test_specs_from_ds`](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/swebench_utils/test_spec.py#L173).

This function returns a [`TestSpec`](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/swebench_utils/test_spec.py#L30) object, which contains all information necessary for environment setup (install commands, test commands, etc.). The detailed mapping of environment installation and test commands is specified in

[`python.py`](https://github.com/lblankl/SWE-MiniSandbox/blob/main/SWE-bench/swebench/harness/constants/python.py).

___

## SWE-bench Verified Evaluation with MiniSandbox

Once the environment cache has been prepared, you can run evaluation with SWE-MiniSandbox.

### Direct Model Evaluation

First, serve your model. For example:

```bash
modelp=SWE-bench/SWE-agent-LM-32B

conda activate vllm   # use a conda env with vllm installed
vllm serve "$modelp" \
    --tensor-parallel-size 8 \
    --async-scheduling \
    --served-model-name custom
```

Then, run the evaluation. This command runs SWE-agent over the dataset and produces a `results.json` file that includes the resolved rate:

```bash
# Same setup (env vars, httpbin, etc.) as in the cache preparation section

sweagent run-batch --config "$basedir/SWE-MiniSandbox/config/sweagent_infer_default.yaml" \
    --env_type sandbox \
    --instances.deployment.conda_env="$env_dir" \
    --agent.model.api_base http://0.0.0.0:8000/v1 \
    --agent.model.name custom \
    --agent.eval True \
    --random_delay_multiplier=1 \
    --instances.deployment.root_base="$sandbox_dir" \
    --instances.deployment.tool_path="$basedir/SWE-MiniSandbox/SWE-agent/tools" \
    --instances.deployment.git_base_path="$cached_git" \
    --instances.deployment.shared_venv="$shared_venv_dir" \
    --output_dir "$output_dir" \
    --instances.database "$database" \
    --instances.start 0 \
    --instances.end 500 \
    --num_workers 10

kill $HTTPBIN_PID  # do not forget to kill the httpbin server

python "$basedir/SWE-MiniSandbox/sh/eval.py" \
    --pred_file_path "$output_dir/results.json"

# The resolved rate will be saved in $output_dir/results.json
```

### Evaluation from Patch Submissions

If you already have a set of patch submissions (e.g., from a previous run or a different model), you can evaluate them directly. Suppose your patch file is saved as `$output_dir/preds.json` (this is usually produced by `sweagent run-batch`).

Run:

```bash
# Same setup (env vars, httpbin, etc.) as before

sweagent run-batch --config "$basedir/SWE-MiniSandbox/config/sweagent_infer.yaml" \
    --env_type sandbox \
    --instances.deployment.conda_env="$env_dir" \
    --agent.model.api_base http://0.0.0.0:8000/v1 \
    --random_delay_multiplier=1 \
    --instances.deployment.root_base="$sandbox_dir" \
    --instances.deployment.tool_path="$basedir/SWE-MiniSandbox/SWE-agent/tools" \
    --instances.deployment.git_base_path="$cached_git" \
    --instances.deployment.shared_venv="$shared_venv_dir" \
    --output_dir "$output_dir" \
    --instances.database "$database" \
    --instances.model_patch_file "$output_dir/preds.json"

kill $HTTPBIN_PID  # do not forget to kill the httpbin server
python "$basedir/SWE-MiniSandbox/sh/eval.py" \
    --pred_file_path "$output_dir/results.json"
```


This will apply the patches from `preds.json` to the cached environments, run the tests, and produce evaluation results in the output directory.

