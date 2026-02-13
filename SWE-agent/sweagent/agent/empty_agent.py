from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
import subprocess
import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator
from simple_parsing.helpers.fields import field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from tenacity import RetryError
from typing_extensions import Self
from unidiff import UnidiffParseError

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    HumanModel,
    HumanThoughtModel,
    InstanceStats,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.agent.reviewer import (
    ChooserRetryLoop,
    RetryLoopConfig,
    ReviewSubmission,
    ScoreRetryLoop,
    get_retry_loop_from_config,
)
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    TotalCostLimitExceededError,
)
from sweagent.tools.parsing import (
    ActionOnlyParser,
    ThoughtActionParser,
)
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _convert_paths_to_abspath, _strip_abspath_from_dict
from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from sweagent.utils.log import get_logger
from sweagent.utils.patch_formatter import PatchFormatter

from sweagent.agent.agents import AbstractAgent,TemplateConfig,DefaultAgentConfig,EmptyAgentConfig
from openai import OpenAI
RETRY_WITH_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITH-OUTPUT###"
RETRY_WITHOUT_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITHOUT-OUTPUT###"
EXIT_FORFEIT_TOKEN = "###SWE-AGENT-EXIT-FORFEIT###"
class _BlockedActionError(Exception):
    """Raised when the agent's action is blocked"""


class _RetryWithOutput(Exception):
    """Used for internal control flow"""


class _RetryWithoutOutput(Exception):
    """Used for internal control flow"""


class _ExitForfeit(Exception):
    """Used for internal control flow"""


class _TotalExecutionTimeExceeded(Exception):
    """Used for internal control flow"""


# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me a joke."},
#     ]
# )
class EmptyAgent(AbstractAgent):
    """This EmptyAgent implementation serves as environment constructor and validator for
        SWE-MiniSandbox environments. It does not perform any actions. Only when the state of the instance is marked
        as "passed", the environment is considered successfully prepared.
    """
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        # Set in run method
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0

    @classmethod
    def from_config(cls, config: EmptyAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self._chook.add_hook(hook)

    # Properties
    # ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):
        # Do import here to avoid circular dependency
        from sweagent.run.run_single import RunSingleConfig

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset,
        processed through all history processors.
        """
        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]  # type: ignore

        # Chain the history processors
        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages  # type: ignore

    # Methods
    # -------

    def _append_history(self, item: dict[str, Any]) -> None:
        """Adds an item to the history."""
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the empty agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """
        print('setup')
        output_dir.mkdir(parents=True, exist_ok=True)


        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        # Save/reset some attributes
        self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)

        self._chook.on_tools_installation_started()
        # self.tools.install(self._env)
        self._chook.on_setup_attempt()
        # self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        assert self._env is not None
        assert self._problem_statement is not None
        self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self.add_system_message_to_history()
        self.add_demonstrations_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(self._env))
        self._chook.on_setup_done()

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def add_demonstrations_to_history(self) -> None:
        """Add demonstrations to history"""
        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:
        """Load demonstration from disk and add to history"""
        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)

        # Load history
        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(_demo_text)["history"]
        else:
            demo_history = json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:
            # Add demonstrations to history step-by-step
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:
            # Add demonstration as single message to history
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates

        Args:
            **kwargs: additional keyword arguments to be added to the format dictionary
        """
        assert self._problem_statement is not None
        assert self._env is not None
        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state)
        and add to history.

        Args:
            templates: templates to populate and add to history
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        format_dict["working_dir"] = '/'+self._env.deployment.git_folder
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            # Show standard output template if there is observation content
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        """
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)  # type: ignore

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None
        attempt_data["environment"] = self._env.name
        return attempt_data

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_model_requery_history(
        self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:
        """Ask the model to correct after a hitting one of the following errors:

        1. Malformatted output (could not parse action)
        2. Blocked action (command is on the blocklist)
        3. Bash command syntax error

        At the time this function is called, the proposed action and observation are not part of the history
        yet.

        This function adds temporary history based on the error template and queries the model.
        If the model is able to correct itself, the records of the mistakes will not be part of the history
        (but they are saved in the trajectory).

        Args:
            error_template: error template
            output: model output
            **kwargs: keyword arguments to be passed to the error template

        Returns:
            model output after requery
        """
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")

        return self.messages + [
            {"role": "assistant", "content": output, "agent": self.name, "message_type": "assistant"},
            {"role": "user", "content": error_template, "agent": self.name, "message_type": "user"},
        ]

    def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:
        """For most exceptions, we attempt to still extract the patch and submit that.
        This means we send the `submit` command to the runtime and parse the output.
        """
        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not self._env.deployment.is_alive(timeout=10):
            # The agent is dead. This is very bad. Maybe we can take a 'diff' that was saved
            # for a previous step? (if running with diff in tools)
            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step
        #Let us manually run the submission command and collect the output
        
        
        step = self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        return step

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it.

        Args:
            step:
            observation: If specified, will use this rather than stepobservation
            force_submission: If True, will always submit even if no submission is found

        Returns:
            step: step with submission and observation updated (if submission was found)
        """
        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            try:
                self._env.get_patch('/res.patch')
                submission = self._env.read_file("/res.patch", encoding="utf-8", errors="backslashreplace")
                #start evaluation
                reward,p2p,f2p,output=self._env._calculate_reward(p2p=-1,f2p=1)
                # reward,p2p,f2p,output=0,{},{},''
                self.info['reward']=reward
                self.info['test_out']=output
                self.info['p2p']=p2p
                self.info['f2p']=f2p
                #self.info['reward']=1
            except FileNotFoundError:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            except Exception as e:
                raise e
                # self.logger.exception("Failed to read submission file, got %s", e)
                # step.done = True
                # return step

            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:
        """Get the edited files with context from the patch"""
        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: self._env.read_file(  # type: ignore[attr-defined]
                            PurePosixPath("/") / self._env.repo.repo_name / path  # type: ignore[attr-defined]
                        ),
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None
            # We still need to populate the variables
        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell
            output: output from model (only used for error handling)

        Returns:
            action_execution_output: action execution output
        """
        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            # step.state = self.tools.get_state(env=self._env)  # for history
            step.state={}
            return step

        assert self._env is not None
        self._chook.on_action_started(step=step)
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        except CommandTimeoutError:

            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception as f:
                self.logger.exception("Failed to interrupt session after command timeout: %s", f, exc_info=True)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise

            
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        self._chook.on_action_executed(step=step)
        step.state = self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        return self.handle_submission(step)

   
    def forward(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors.

        All exceptions raised will contain the `StepOutput` object
        with some of the attributes set.

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        # we continuously add actions, output etc. to the step object
        # because some of the specific exception handling requires some of these
        # attributes (e.g., if we want to requery the model for a bash syntax error, we
        # need to have the previous model output to format the requery template)
        step = StepOutput()
        step.query = copy.deepcopy(history)
        try:
            # Forward model and get actions
            self._chook.on_model_query(messages=history, agent=self.name)
            # todo: Add all options to the extra info
            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=history,
                )
                output = best.completion
                # todo: Handle history and trajectory
                step.extra_info.update(best.extra_info)
            else:
                output = self.model.query(history)  # type: ignore
            step.output = output["message"]
            # todo: Can't I override the parser in __init__?
            step.thought, step.action = self.tools.parse_actions(output)
            step.thinking_blocks = output.get("thinking_blocks", [])
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n🎬 ACTION\n{step.action.strip()}")
            self._chook.on_actions_generated(step=step)
            return self.handle_action(step)
        except Exception as e:
            if step.action == step.thought == "":
                # Probably the parsing failed/no action included. Let's still fill in thought
                # so that trajectory viewers have something to show us for this step.
                step.thought = step.output
            # Attach the step object to the exception
            e.step = step  # type: ignore
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can.
        For example, if the model outputs a bash command that has syntax errors,
        we will not execute it but requery the model for a corrected command.

        Note: This will update the trajectory, but not the history.

        Args:
            history: history to forward

        Returns:
            step_output: step output
        """

        def handle_error_with_autosubmission(exit_status: str, message: str) -> StepOutput:
            """Attempts to autosubmit (extract patch from the environment) and stops the loop."""
            self.logger.warning(message)
            return self.attempt_autosubmission_after_error(
                StepOutput(
                    thought=message,
                    exit_status=exit_status,
                    output=message,
                    done=True,
                )
            )

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[dict[str, str]]:
            """Requeries the model if the error is a format/blocklist/bash syntax error."""
            self.logger.warning("Requerying model after %s (%dth requery)", type(exception).__name__, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)

            # Errors that are raised

            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

            # Errors that cause requery

            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails
                )
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
                # Try if simply resampling helps here
                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass
                # Requery with the same template as the last step

            # Errors that cause exit

            except _ExitForfeit:
                self.logger.info("Exiting due to forfeit")
                return handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                )

            except _TotalExecutionTimeExceeded:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                )

            except CommandTimeoutError:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                )

            except ContextWindowExceededError:
                return handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                )
            except TotalCostLimitExceededError:
                raise
            except CostLimitExceededError:
                return handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "query": step.query,
                "extra_info": step.extra_info,
            },
        )
        self.trajectory.append(trajectory_step)

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        self.info["model_stats"] = self.model.stats.model_dump()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output
    def process_precheck(self,reward,f2p,p2p,out):
        """Process the precheck results and update the agent info accordingly."""
        self.info = AgentInfo()
        
        self.info['reward']=reward
        if reward==1:
            self.info['test_out']=out
            self.info['p2p']=p2p
            self.info['f2p']=f2p
            self.info['exit_status']='passed'
        else:
            self.info['test_out']=out
            self.info['p2p']=p2p
            self.info['f2p']=f2p
            self.info['exit_status']='failed'
            
    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the EmptyAgent in the given environment with the given problem statement.
        It first performs the pre-checks to validate the environment, then record all \
        the results from the pre-checks and save them to the trajectory file.

        Attributes:
            env: environment to run the agent in
            problem_statement: problem statement to solve
            output_dir: directory to save the trajectory and other outputs
        """
        env._chook.on_start_deployment()
        self.process_precheck(*env.pre_check())
        env._chook.on_pre_check_finish()
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        # Run action/observation loop
        self._chook.on_run_start()
        # step_output = StepOutput()
        # while not step_output.done:
        #     step_output = self.step()
        #     self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        # data = self.get_trajectory_data()
        return AgentRunResult(info=self.info, trajectory=self.trajectory)
