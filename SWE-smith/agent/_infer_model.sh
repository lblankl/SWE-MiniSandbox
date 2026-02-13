#!/bin/bash

sweagent run-batch --config agent/swesmith_infer.yaml \
	--instances.deployment.docker_args=--memory=10g \
	--agent.model.api_base https://svt25nwvnpipwz.r20.modal.host/v1 \
	--random_delay_multiplier=1 \
	--output_dir trajectories/john-b-yang/swesmith.ablation.bug.lm_reimplement_500
