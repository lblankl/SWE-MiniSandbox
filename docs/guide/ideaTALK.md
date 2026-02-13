# Architecture

This page walks through the overall architecture of the SWE-MiniSandbox framework.  
If you just want to run it, please refer to the [Quick Start](quick-start.md) guide.

## SWE-Agent Integration

SWE-MiniSandbox integrates with SWE-Agent to provide a complete solution for batch execution of software engineering tasks.  
For background, we recommend first reviewing the [SWE-Agent Architecture](https://swe-agent.com/latest/background/architecture).

<p align="center">
  <a href="https://swesmith.com/">
    <img src="architecture.png" style="height: 20em" alt="SWE-MiniSandbox Architecture" />
  </a>
</p>

The main flow is:

1. The `sweagent` command line executable initializes an instance of the `SWEsbEnv` class.
2. `SWEsbEnv` is a modification of the original `SWEEnv` class in SWE-Agent and manages the interaction between the agent and the MiniSandbox environments.
   - The original `SWEEnv` class is retained for compatibility with standard Gym environments.
   - We extend `SWEEnv` with additional functionality to support convenient reward computation.
3. `SWEsbEnv` initializes the MiniSandbox deployment, which manages our container-free local Gym environments.
4. The MiniSandbox deployment uses SWE-Rex to manage terminal sessions.
   - We introduce a new runtime class and modify the remote runtime in SWE-Rex to support this project.

In addition, we re-implement the `Agent_sb` class based on the original `Agent` class to support the MiniSandbox-specific agent workflow.  
The original `Agent` class remains available for standard Gym environment interactions.

## Sky-RL Integration

Sky-RL typically uses a user-defined custom generator class to launch agent rollout processes.  
For SWE-MiniSandbox, we re-implement the generator class following the mini-sweagent example.

<p align="center">
  <a href="https://swesmith.com/">
    <img src="SkyRL.png" style="height: 15em" alt="Sky-RL Integration" />
  </a>
</p>

The `SweAgentGenerator` class launches multiple `init_and_run_sb` or `init_and_run_container` processes to perform agent rollouts in parallel.  
Each process initializes an instance of `SWEsbEnv` or `SWEEnv` to manage interactions between the agent and the respective environments.
