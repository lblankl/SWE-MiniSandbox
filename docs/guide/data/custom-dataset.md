# Custom Dataset Support

Currently we support two datasets: SWE-bench Verified and SWE-smith. To support other datasets, for example, SWE-Gym (We will support it in the future), we also provide a tutorial here.

## Dataset Format Alignment
To support custom datasets, temporarily we recommend to use smith pipeline. So you need to make sure that your dataset follows a similar format as SWE-smith.
It should have:
```yaml
{
  "instance_id": str,          # Unique identifier for each instance
  "patch": str,                # The reversed golden patch for the instance
  "FAIL_TO_PASS": list[str],   # The list of f2p cases for the instance
  "PASS_TO_PASS": list[str],   # The list of p2p cases for the instance
   "image_name": str,        # The image name for the instance, instance with the same image_name share the same venv cache (see Data Preparation guide of swe-smith)
    "repo": str,              # The github repo id for the instance, used for github repo cache and venv cache path mapping
    "problem_statement": str, # The problem statement of the instance
    "test_patch": str,         # The test patch for the instance
    "base_commit": str,      # The base commit hash for the instance, if not provided, we use instance_id by default
    "python_version": str,   # The python version for the instance, used for venv creation, if not provided, use 3.11 by default
}
```
Note that the `patch` field is necessary for environment validation. And for swesmith, it should be the reversed patch to introduce bugs. When we validate the environment, we will apply the patch reversely to fix the bug. But for swebench, the patch is the normal patch to fix the bug. (See in our logic [here](https://github.com/lblankl/SWE-MiniSandbox/blob/main/SWE-agent/sweagent/environment/swe_sbenv.py#L170)).

The `base_commit` field is used to checkout the repo to a specific commit before applying the patches. If not provided, we will use the instance_id as the commit hash. You can choose to provide this field if you want to use a specific commit for the instance.

The `test_patch` field does not exist in swe-smith dataset, which means in smith tasks, all the test cases are visiable to the agent. This is different from swe-bench verified dataset, where only the test cases introduced before the fixing of the bug are visible to the agent. 
However, in custom datasets, you can choose to provide the `test_patch` field to hide some test cases from the agent (The base_commit should not contain those cases). The `test_patch` is the patch that introduces all the test cases. During environment validation, we will first apply the `test_patch` to introduce all the test cases, and then apply the `patch` reversely to fix the bug. 

The `python_version` field is used to create the venv based on the corresponding python version. If not provided, we will use python 3.11 by default.

With the above format, you can easily adapt your custom dataset to our framework following the [Smith Environment Preparation](swe-smith.md) guide.
Smith pipeline is designed for many to one mapping from instance_id to image_name.
For one to one mapping from instance_id to image_name, like SWE-Gym, you can simplify this process accordingly like [SWE-bench Environment Preparation](swe-bench.md) guide.

## Installation and Test Scripts Mapping
You need to define the mapping from instance_id to [installation commands](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/customer_instance.py#L1) and
[test commands](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/customer_instance.py#L16) for your custom dataset. 

For how to define the installation commands and test commands, you can refer to the existing implementations for SWE-smith [get_install_commands](https://github.com/lblankl/SWE-MiniSandbox/blob/main/R2E-Gym/src/r2egym/swesmith/utils.py#L100) and [get_test_command](https://github.com/lblankl/SWE-MiniSandbox/blob/main/R2E-Gym/src/r2egym/swesmith/utils.py#L108).
The reward calculation function is [here](https://github.com/lblankl/SWE-MiniSandbox/blob/main/sandboxdev/swesandbox/sandbox_deployment.py#L914).