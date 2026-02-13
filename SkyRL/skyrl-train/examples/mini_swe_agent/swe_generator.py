import asyncio
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import ray
from pathlib import Path
from sweagent.agent.skyagent.agents import get_agent_from_config
from sweagent.environments.skyrl.swe_env import SWEEnv

from sweagent.agent.skyagent.agents import DefaultAgent

from sweagent.run.run_single import RunSingleConfig
from sweagent.run.common import save_predictions


from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)
from sweagent.run.batch_instance import BatchInstance
from sweagent.run.run_batch import RunBatchConfig
from sweagent.run.common import BasicCLI
class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        #output = self.execute_action(self.parse_action(response))
        #observation = self.render_template(self.config.action_observation_template, output=output)
        output = response["content"]
        thought, action = self.tools.parse_actions(output)
        run_action: str = self.tools.guard_multiline_input(action).strip()
        observation = self._env.communicate(
                input = run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: BatchInstance,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: DictConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger
    agent_config = sweagent_config.get("agent", {})
    
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    agent_config.model.update(sampling_params)
    
    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    

    output_dir = Path(sweagent_config.output_dir)/ f"step_{global_step}" / training_phase / instance.problem_statement.id
    output_dir.mkdir(parents=True, exist_ok=True)

    single_run_replay_config = RunSingleConfig(
            agent=agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
    (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
        )
    agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
    instance.env.name = f"{instance.problem_statement.id}"
    try:
        # env = get_sb_environment(sweagent_config, instance, data_source)
        bundles = agent_config.tools.bundles
        env = SWEEnv.from_config(ds=instance.ds,bundles=bundles,config=sweagent_config.env)
        env.start()
        #agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        agent = get_agent_from_config(agent_config)
        result = agent.run(
                problem_statement=instance.problem_statement,
                env=env,
                output_dir=output_dir,
            )
        #exit_status, result = agent.run(instance.problem_statement)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance.instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        env.close()
    save_predictions(output_dir, instance.problem_statement.id, result)
    #     # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
       
    #     # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
    #     instance_id = instance.instance_id
    #     filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
    #     path = path / filename
        # if agent is not None:
        #     eval_error = None
        #     try:
        #         result = evaluate_trajectory(instance, result, sweagent_config, data_source)
        #         reward = int(result["resolved"])
        #         eval_error = result["eval_error"]
        #         if eval_error:
        #             error = eval_error
        #             logger.debug(f"Error during evaluation {eval_error}")
        #     except Exception as e:
        #         logger.debug(f"Error during evaluation {e}")
        #         logger.debug(f"traceback: {traceback.format_exc()}")
        #         eval_error = str(e)
        #         error = str(e)

        #     save_traj(agent, path, exit_status=exit_status, result=result, extra_info=extra_info, reward=reward, eval_error=eval_error)  # type: ignore[arg-type]
    info = result.info if result is not None else {}
    reward = info.get("reward", reward)
    error = "error"
    return (agent.history if agent is not None else [], reward, error)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)

        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        sweagent_config: DictConfig,
        instance: List[BatchInstance],
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error = await init_and_run.remote(
            instance,
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            return None, None, None, None, None, None

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        response_ids: List[int] = []
        loss_mask: List[int] = []

        # We remove trailing `user` messages - this is added by Mini-SWE-Agent to capture the final git diff for the trajectory
        last_idx = len(response_messages) - 1
        while response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Please ensure that your environment is configured correctly and the `OPENAI_BASE_URL` points to the HTTP server from the inference engine client"
            )
        response_messages = response_messages[: last_idx + 1]

        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = encode_messages_subset([message], self.tokenizer)

            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)

            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
        # Extract prompt ids
        prompt_ids = initial_input_ids

        # Calculate maximum response tokens allowed
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        tasks = []
        
        datasets=[]
        for i in range(len(env_extras)):
            data_instance = env_extras[i]["instance"]
            data_instance['instance_id'] = trajectory_ids[i]
            datasets.append(data_instance)

        cli_cfg = self.generator_cfg.sweagent_cfg
        # turn cli_cfg into args
        args = []
        for key, value in cli_cfg.items():
            args.append(f"--{key}")
        
            args.append(str(value))
        sweagent_config = BasicCLI(RunBatchConfig, help_text='s').get_config(args)
        sweagent_config.instances.dataset = datasets
        instances = sweagent_config.instances.get_instance_configs()
        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    sweagent_config,
                    instances[i],
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        # Filter out the `None` entries, which means that trajectory generation failed
        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]
        if not len(responses):
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
