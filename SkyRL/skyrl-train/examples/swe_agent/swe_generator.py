import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import copy
import torch
from sweagent.types import AgentInfo, AgentRunResult, TrajectoryStep
import ray
import shutil
from collections.abc import Mapping
from pathlib import Path
from sweagent.agent.agents import AgentConfig, get_agent_from_config
from sweagent.environment.swe_env import SWEEnv
from sweagent.environment.swe_sbenv import SWEsbEnv

from sweagent.run.run_single import RunSingleConfig
from sweagent.run.common import save_predictions
from skyrl_train.utils.trainer_utils import validate_generator_output

from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    encode_messages_subset,
)
from sweagent.run.batch_instances import BatchInstance
from sweagent.run.run_batch import RunBatchConfig
from sweagent.run.common import BasicCLI
import json
import time
import psutil
def get_total_cpu_percent(proc: psutil.Process) -> float:
    # 当前进程 + 所有子进程的 CPU 使用率相加
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0.0

    total = 0.0
    for p in procs:
        try:
            total += p.cpu_percent(interval=None)
        except psutil.Error:
            continue
    return total

def monitor_cpu_with_children(func, *args, interval=0.5, **kwargs):
    proc = psutil.Process()
    cpu_samples = []

    # 预热
    get_total_cpu_percent(proc)

    start_time = time.time()
    result = None
    finished = False

    def run_target():
        nonlocal result, finished
        result = func(*args, **kwargs)
        finished = True

    import threading
    t = threading.Thread(target=run_target)
    t.start()

    while not finished:
        time.sleep(interval)
        total_cpu = get_total_cpu_percent(proc)
        # 转成“核数”
        cores = total_cpu / psutil.cpu_count()
        cpu_samples.append((time.time(), cores))

    t.join()
    end_time = time.time()
    return result, cpu_samples, start_time, end_time



@ray.remote(num_cpus=0.01,resources={"container_start_up": 1})
def start_container_remote(instance: BatchInstance):
    return SWEEnv.from_config(config=instance.env,ds=instance.ds)

@ray.remote(num_cpus=0.01,resources={"sandbox_start_up": 1})
def start_sandbox_remote(instance: BatchInstance,bundles):
    return SWEsbEnv.from_config(ds=instance.ds,bundles=bundles,config=instance.env)
@ray.remote(num_cpus=0.01,resources={"container": 1})
def init_and_run_container_remote(
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
    return init_and_run_container(
        instance,
        litellm_model_name,
        sweagent_config,
        generator_cfg,
        data_source,
        sampling_params,
        trajectory_id,
        global_step,
        training_phase,
    )



@ray.remote(num_cpus=0.01,resources={"sandbox": 1})
def init_and_run_sb_remote(
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
    return init_and_run_sb(
        instance,
        litellm_model_name,
        sweagent_config,
        generator_cfg,
        data_source,
        sampling_params,
        trajectory_id,
        global_step,
        training_phase,
    )

def init_and_run_container(
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
    """Initialize and run the container agent loop for the given instance.

    Attributes:
        instance: The BatchInstance to run.
        litellm_model_name: Deprecated. Not used.
        sweagent_config: One instance of RunBatchConfig of SWE-agent project 
        generator_cfg: Deprecated. Not used.
        data_source: Deprecated. Not used.
        sampling_params: The sampling parameters to use for the model.
        trajectory_id: The trajectory ID. Deprecated. Not used.
        global_step: The global step. Used for output directory structure.
        training_phase: The training phase. Used for output directory structure.
    """
    from loguru import logger
    agent_config = sweagent_config.agent
    
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    # agent_config.model.update(sampling_params)
    agent_config.model = agent_config.model.model_copy(update=sampling_params)
    agent_config.model.completion_kwargs=sampling_params
    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    
    env_type=sweagent_config.env_type
    

    single_run_replay_config = RunSingleConfig(
            agent=agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
    
    
    instance.env.name = f"{instance.problem_statement.id}"
    
    # implement loop retry logic
    successful=False
    max_retries = 20
    num_retries = 0

    time_records={}
    start_time= time.time()
    while num_retries < max_retries:
        num_retries += 1
        try:
            output_dir = Path(sweagent_config.output_dir)/ f"step_{global_step}" / training_phase / instance.problem_statement.id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            # env = get_sb_environment(sweagent_config, instance, data_source)
            bundles = agent_config.tools.bundles
            # env = SWEEnv.from_config(ds=instance.ds,bundles=bundles,config=instance.env)
            
           
            
            
            def start_env_logic(instance):
                env = ray.get(start_container_remote.remote(instance))
                Time_data = env.start()
                return env, Time_data
            env = ray.get(start_container_remote.remote(instance))
            env_startup_time = time.time()
            Time_data = env.start()
            env_ready_time = time.time()
            # (env, Time_data), cpu_samples, env_startup_time, env_ready_time = monitor_cpu_with_children(start_env_logic, instance)
            
            (output_dir / f"after_env_init.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            
            time_records[num_retries]={'env_start_time':env_startup_time,'env_ready_time':env_ready_time,'time_taken':env_ready_time - env_startup_time}
            time_records[num_retries]['detailed_time_data']=Time_data
            try:
                rollout_signal = ray.get_actor("rollout_signal_actor")  # 或从 self.rollout_signal 拿
            except ValueError:
                rollout_signal = None

            if rollout_signal is not None:
                # 调用 async 远程方法 wait_for_step(step_id)，并在当前任务里阻塞等待
                ray.get(rollout_signal.wait_for_step.remote(global_step))

            (output_dir / f"after_rollout_signal.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            #agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
            agent = get_agent_from_config(agent_config)
            agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
            agent_start_time= time.time()

            (output_dir / f"startagent.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            result = agent.run(
                    problem_statement=instance.problem_statement,
                    env=env,
                    output_dir=output_dir,
                )
            agent_end_time= time.time()
            time_records[num_retries].update({'agent_start_time':agent_start_time,'agent_end_time':agent_end_time,'agent_time_taken':agent_end_time - agent_start_time})
            if len(agent.history)<=2:
                raise Exception("Agent history too short, likely failed run.")
            successful=True
            break  # Exit the retry loop if successful
            #exit_status, result = agent.run(instance.problem_statement)  # type: ignore[arg-type]
        except Exception as e:
            env_ready_time= time.time()
            time_records[num_retries]={'env_start_time':env_startup_time,'env_ready_time':env_ready_time,'time_taken':env_ready_time - env_startup_time}
            #output_dir = Path(self.output_dir) / instance.problem_statement.id
            #write the exception to a file
            (output_dir / f"exception_{num_retries}.log").write_text(traceback.format_exc())
            #remove the output dir to avoid partial results
            # output_dir.rmdir()
            # if output_dir.exists() and num_retries<max_retries:
            #     shutil.rmtree(output_dir, ignore_errors=True)
            logger.error("Error processing instance {}: {}", instance.problem_statement.id, e, exc_info=True)
            #sleep for a while before retrying
            time.sleep(1)

            exit_status, result = type(e).__name__, None
            error = str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            try:
                env.close()
            except Exception as e:
                print("fail to close env")
    end_time= time.time()
    time_records['total_time']={'start_time':start_time,'end_time':end_time,'time_taken':end_time - start_time}
    if successful:
        save_predictions(output_dir, instance.problem_statement.id, result)
        #also write time records
    (output_dir / f"time_records.yaml").write_text(
        yaml.dump(time_records, indent=2)
        )
    
    info = result.info if result is not None else {}
    reward = info.get("reward", reward)
    error = "error"
    return (agent.history if agent is not None else [], reward, error)


def init_and_run_sb(
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
    """Initialize and run the sandbox agent loop for the given instance.
    Similar to init_and_run_container but uses minisandbox environment.
    """
    from loguru import logger
    agent_config = sweagent_config.agent
    
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    # agent_config.model.update(sampling_params)
    agent_config.model = agent_config.model.model_copy(update=sampling_params)
    agent_config.model.completion_kwargs=sampling_params
    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    
    env_type=sweagent_config.env_type
    

    single_run_replay_config = RunSingleConfig(
            agent=agent_config,
            problem_statement=instance.problem_statement,
            env=instance.env,
        )
    
    
    instance.env.name = f"{instance.problem_statement.id}"
    
    # implement loop retry logic
    successful=False
    max_retries = 10
    num_retries = 0
    time_records={}
    Cpu_data_records={}
    start_time= time.time()
    while num_retries < max_retries:
        num_retries += 1
        try:
            output_dir = Path(sweagent_config.output_dir)/ f"step_{global_step}" / training_phase / instance.problem_statement.id
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{instance.problem_statement.id}.config.yaml").write_text(
            yaml.dump(single_run_replay_config.model_dump_json(), indent=2)
            )
            # env = get_sb_environment(sweagent_config, instance, data_source)
            bundles = agent_config.tools.bundles
            # env = SWEEnv.from_config(ds=instance.ds,bundles=bundles,config=instance.env)
            # env_startup_time= time.time()
            # #
            # env = ray.get(start_sandbox_remote.remote(instance,bundles))
            # #env = SWEsbEnv.from_config(ds=instance.ds,bundles=bundles,config=instance.env)
           
            # Time_data=env.start()
            # env_ready_time= time.time()
            def start_env_logic(instance):
                env = ray.get(start_sandbox_remote.remote(instance,bundles))
                Time_data = env.start()
                return env, Time_data

            (env, Time_data), cpu_samples, env_startup_time, env_ready_time = monitor_cpu_with_children(start_env_logic, instance)
            Cpu_data_records[num_retries] = cpu_samples
            time_records[num_retries]={'env_start_time':env_startup_time,'env_ready_time':env_ready_time,'time_taken':env_ready_time - env_startup_time}
            time_records[num_retries]['detailed_time_data']=Time_data
            try:
                rollout_signal = ray.get_actor("rollout_signal_actor")  # 或从 self.rollout_signal 拿
            except ValueError:
                rollout_signal = None

            if rollout_signal is not None:
                # 调用 async 远程方法 wait_for_step(step_id)，并在当前任务里阻塞等待
                ray.get(rollout_signal.wait_for_step.remote(global_step))
            #agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
            agent = get_agent_from_config(agent_config)
            agent.replay_config = single_run_replay_config  # type: ignore[attr-defined]
            agent_start_time= time.time()
            result = agent.run(
                    problem_statement=instance.problem_statement,
                    env=env,
                    output_dir=output_dir,
                )
            agent_end_time= time.time()


            time_records[num_retries].update({'agent_start_time':agent_start_time,'agent_end_time':agent_end_time,'agent_time_taken':agent_end_time - agent_start_time})
            
            if len(agent.history)<=2:
                raise Exception("Agent history too short, likely failed run.")
            successful=True

            break  # Exit the retry loop if successful
            #exit_status, result = agent.run(instance.problem_statement)  # type: ignore[arg-type]
        except Exception as e:
            env_ready_time= time.time()
            time_records[num_retries]={'env_start_time':env_startup_time,'env_ready_time':env_ready_time,'time_taken':env_ready_time - env_startup_time}
            #output_dir = Path(self.output_dir) / instance.problem_statement.id
            #write the exception to a file
            (output_dir / f"exception_{num_retries}.log").write_text(traceback.format_exc())
            #remove the output dir to avoid partial results
            # output_dir.rmdir()
            # if output_dir.exists() and num_retries<max_retries:
            #     shutil.rmtree(output_dir, ignore_errors=True)
            logger.error("Error processing instance {}: {}", instance.problem_statement.id, e, exc_info=True)
            
            # einfo=AgentInfo()
            # etra=[TrajectoryStep()]
            exit_status, result = type(e).__name__, None
            error = str(e)
            extra_info = {"traceback": traceback.format_exc()}
        finally:
            try:
                env.close()
            except Exception as e:
                print("fail to close env")
    end_time= time.time()
    time_records['total_time']={'start_time':start_time,'end_time':end_time,'time_taken':end_time - start_time}
    if successful:
        save_predictions(output_dir, instance.problem_statement.id, result)
        #also write time records
    (output_dir / f"time_records.yaml").write_text(
        yaml.dump(time_records, indent=2)
        )
    # write cpu records to json cpu_usage_records.json
    
    (output_dir / f"cpu_usage_records.json").write_text(
        json.dumps(Cpu_data_records, indent=2)
    )

    
    info = result.info if result is not None else {}
    reward = info.get("reward", reward)
    error = "error"
    return (agent.history if agent is not None else [], reward, error)

def flatten_dict(d: Mapping, parent_key: str = "", sep: str = "."):
    """
    输入:
      {"agent": {"type": "skydefault", "model": {"name": "xxx"}},
       "output_dir": "/path"}
    输出:
      {"agent.type": "skydefault",
       "agent.model.name": "xxx",
       "output_dir": "/path"}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    
    return items
class SweAgentGenerator(SkyRLGymGenerator):
    """Customized SkyRLGymGenerator for SWE-Agent."""
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
        cli_cfg = self.generator_cfg.sweagent
        # # turn cli_cfg into args
       
        if hasattr(cli_cfg, "to_container"):
            cli_cfg = cli_cfg.to_container(resolve=True)
        flat_cfg = flatten_dict(cli_cfg)    
        args = []
        for key, value in flat_cfg.items():
            args.append(f"--{key}")
        
            args.append(str(value))
        self.sweagent_config = BasicCLI(RunBatchConfig, help_text='s').get_config(args)
       
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("SweAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        sweagent_config: DictConfig,
        instance: BatchInstance,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        """
         The rollout inner loop for mini-swe-agent. It cakls the init_and_run_container_remote or init_and_run_sb_remote based on the env_type in sweagent_config.
        
        Attributes:
            sweagent_config: The sweagent configuration. RunBatchConfig object.
            instance: The BatchInstance to run.
            prompt: The input prompt. Deprecated. Not used.
            env_extras: Extra environment information.
            max_tokens: The maximum number of tokens to generate.
            max_input_length: The maximum input length.
            sampling_params: The sampling parameters to use for the model.
            trajectory_id: The trajectory ID. Deprecated. Not used.
            batch_metadata: The batch metadata.
        """
        # sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        if sweagent_config.env_type=='sandbox':
            ref = init_and_run_sb_remote.remote(
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
        elif sweagent_config.env_type=='docker':
            
            ref = init_and_run_container_remote.remote(
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
        else:
            raise Exception()
        messages, reward, error = await asyncio.to_thread(ray.get, ref)
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
        Generate trajectories for the input batch. It call the minisweagent_agent_loop for each instance in the batch concurrently.
        Returns outputs in the same order as the input batch.

        Attributes:
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
            data_instance = copy.deepcopy(env_extras[i]["instance"])
            
            data_instance['traj_id'] = data_instance['instance_id']+'@'+str(trajectory_ids[i].repetition_id)
            data_instance['global_step'] = env_extras[i]['global_step']
            datasets.append(data_instance)

        
        
        instances = self.sweagent_config.instances.get_instance_configs_ds(datasets)
        for i in range(len(prompts)):
            
            tasks.append(
                
                self.minisweagent_agent_loop(
                    self.sweagent_config,
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
@ray.remote
def minisweagent_agent_loop(
        tokenizer,
        generator_cfg,
        sweagent_config: DictConfig,
        instance: BatchInstance,
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
        if sweagent_config.env_type=='sandbox':
            ref = init_and_run_sb_remote.remote(
                instance,
                '',
                sweagent_config,
                generator_cfg,
                env_extras["data_source"],
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        elif sweagent_config.env_type=='docker':
            ref = init_and_run_container_remote.remote(
                instance,
                '',
                sweagent_config,
                generator_cfg,
                env_extras["data_source"],
                sampling_params,
                trajectory_id,
                batch_metadata.global_step,
                batch_metadata.training_phase,
            )
        else:
            raise Exception()
        messages, reward, error = ray.get(ref)
  
        if not len(messages):
            return None, None, None, None, None, None

        # TODO (sumanthrh): This is currently hardcoded for SWEBench with 2 initial messages (system and user).
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in (
                "system",
                "user",
            ), "Expected the first two messages to be system and user messages"

        initial_input_ids = tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
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
            msg_encoding = encode_messages_subset([message], tokenizer)

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
@ray.remote
class SWERolloutWorker:
    def __init__(self, sweagent_config,tokenizer,generator_cfg: DictConfig):
        """
        trainer_state 可以是你需要的那些组件：
        - self.generator
        - tokenizer
        - cfg
        - 等等

        简化起见，这里假设你传入的对象里已经有 generator、all_metrics 等。
        也可以只传 generator 和必要的 config，不必传全 Trainer。
        """
        
        self.sweagent_config = sweagent_config
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.litellm_model_name=""
    # @torch.no_grad()
    # async def generate(self) -> GeneratorOutput:
    #     print("[SWERolloutWorker] generate called, prompts:", len(input_batch["prompts"]))
    #     # 暂时不要调用 _generate，先返回简单数据
    #     return {"ok": True, "n": len(input_batch["prompts"])}
    @torch.no_grad()
    def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # 调用你原来的 generator.generate，不改它的接口和实现
        
        generator_output: GeneratorOutput = self._generate(input_batch)

   

        validate_generator_output(len(input_batch["prompts"]), generator_output)

        return generator_output
   
    

    def _generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
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
            data_instance = copy.deepcopy(env_extras[i]["instance"])
            
            data_instance['traj_id'] = data_instance['instance_id']+'@'+str(trajectory_ids[i].repetition_id)
            data_instance['global_step'] = env_extras[i]['global_step']
            datasets.append(data_instance)

        
        
        instances = self.sweagent_config.instances.get_instance_configs_ds(datasets)
        for i in range(len(prompts)):
            
            tasks.append(
                
                minisweagent_agent_loop.remote(
                    self.tokenizer,
                    self.generator_cfg,
                    self.sweagent_config,
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

        all_outputs = ray.get(tasks)
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

