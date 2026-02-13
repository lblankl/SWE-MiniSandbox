#!/bin/bash

sweagent run-batch --num_workers 20 \
    --instances.deployment.docker_args=--memory=10g \
    --config agent/swesmith_gen_claude.yaml \
    --instances.path /home/john-b-yang/swe-smith/logs/experiments/exp8__ig_orig.json \
    --output_dir trajectories/john-b-yang/swesmith_gen__claude-3.5__t-0.00_p-1.00__c.2.00__exp8__ig_orig_run2 \
    --random_delay_multiplier=1 \
    --agent.model.temperature 0.0

# Remember to set CLAUDE_API_KEY_ROTATION=key1:::key2:::key3
