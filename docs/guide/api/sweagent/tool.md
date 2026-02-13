# Overview of Tools
Since the SWE-Agent-7B and SWE-Agent-32B is trained on the following tools, we only support these tools in the SWE-MiniSandbox framework:

      - path: tools/registry
      - path: tools/edit_anthropic
      - path: tools/submit

Other tools is not ensured to work in the SWE-MiniSandbox framework. However, you can still try to add other tools by yourself. Just follow the instruction in the SWE-agent [Configuring tools](https://swe-agent.com/latest/config/tools/) to add more tools.

The tool path in our project is :

      SWE-agent/
      ├── tools/   # tools for SWE-MiniSandbox
      ├── tool/    # tools for SWE-agent (Same as the original SWE-agent repo)

## MiniSandbox Tools
Named tools under the SWE-agent project directory are modified to work in the SWE-MiniSandbox framework. The following tools are supported:

  - [Registry Tool](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tools/registry/)
  - [Edit Anthropic Tool](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tools/edit_anthropic/)
  - [Submit Tool](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tools/submit/)

### Registry Tool
The Registry Tool is used to manage environment variables for the SWE-Agent. We change its default environment file path to `/roots/.swe-agent-env` because the root directory is mounted as read-only in the MiniSandbox framework. You can find the modified code in [registry.py](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tools/registry/lib/registry.py). You can also specify a custom environment file path by setting the `SWE_AGENT_ENV_FILE` environment variable.

### Edit Anthropic Tool
The Edit Anthropic Tool allows the SWE-Agent to interact with file systems. We change the state path to `/roots/state.json` here (See in [_state_anthropic](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tools/edit_anthropic/bin/_state_anthropic))

### Submit Tool
The Submit Tool is used to submit the task. Original submit script [submit](https://github.com/lblankl/SWE-MiniSandbox/tree/main/SWE-agent/tool/submit/bin/submit) directly generates a patch file and echo `<<SWE_AGENT_SUBMISSION>>` (necessary for detecting submission in SWE-agent). In our implementation, we modify the submit script to only echo `<<SWE_AGENT_SUBMISSION>>`. The true path generation logic is moved to the MiniSandbox deployment implementation. You can find the implementation here:

::: swesandbox.sandbox_deployment.SandboxDeployment.get_patch

## Container Tools
We keep the original tools in the `SWE-agent/tool/` directory for container deployment. These tools are not modified and work the same way as in the original SWE-agent repository.