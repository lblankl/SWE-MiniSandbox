# Overview

This section provides a reference for our modified SWE-Agent API used in the SWE-MiniSandbox framework. It includes details about the main classes and functions that have been extended or modified to support our container-free local Gym environments.

## Modified Project layout

    SWE-agnet/
        sweagent/
            environment/
                ...
                repo.py           # SWE-Agent Repo class (Modified)
                swe_env.py        # SWE-Agent Environment class (Modified)
                swe_sbenv.py      # SWE-MiniSandbox Environment class (New)
            agent/
                ...
                agents.py         # SWE-Agent Agent class (Modified)
                empty_agent.py    # SWE-Agent EmptyAgent class (New)
                sky_agent_sb.py   # SWE-MiniSandbox SkyAgent class (New)
                agent_sandbox.py  # SWE-MiniSandbox Agent class (New)
            run/
                ...
                batch_instances.py # SWE-Agent Batch Instance class (Modified)
                run_batch.py       # SWE-Agent Batch Run class (Modified)
        tools/               # SWE-Agent Tools (Modified Only for MiniSandbox)
        tool/                # SWE-Agent Tool (New, Exactly same as the original SWE-Agent)   
            ...