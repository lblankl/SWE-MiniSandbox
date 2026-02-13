import logging
import shlex
import subprocess
import time
import uuid
from swerex.runtime.abstract import BashAction,UploadRequest
from typing import Any
import os
import tempfile
import asyncio
from typing_extensions import Self
from r2egym.repo_analysis.execution_log_parser import parse_log_fn
from swerex import PACKAGE_NAME, REMOTE_EXECUTABLE_NAME
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import DockerDeploymentConfig
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError, DockerPullError
from swerex.runtime.abstract import IsAliveResponse
from swerex.runtime.config import RemoteRuntimeConfig
from swerex.runtime.remote import RemoteRuntime
from swerex.utils.free_port import find_free_port
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive
from r2egym.swesmith.utils import get_test_command,get_install_commands
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    FAIL_ONLY_REPOS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
__all__ = ["DockerDeployment", "DockerDeploymentConfig"]


def _is_image_available(image: str, runtime: str = "docker") -> bool:
    try:
        subprocess.check_call(
            [runtime, "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _pull_image(image: str, runtime: str = "docker") -> bytes:
    try:
        return subprocess.check_output([runtime, "pull", image], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # e.stderr contains the error message as bytes
        raise subprocess.CalledProcessError(e.returncode, e.cmd, e.output, e.stderr) from None


def _remove_image(image: str, runtime: str = "docker") -> bytes:
    return subprocess.check_output([runtime, "rmi", image], timeout=30)


class DockerDeployment(AbstractDeployment):
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        
        self._config = DockerDeploymentConfig(**kwargs)
        self._runtime: RemoteRuntime | None = None
        self._container_process = None
        self._container_name = None
        self.logger = logger or get_logger("rex-deploy")
        self._runtime_timeout = 0.15
        self._hooks = CombinedDeploymentHook()

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: DockerDeploymentConfig) -> Self:
        return cls(**config.model_dump())

    def apply_test_patch(self,patch):
        """Applies a patch to the testbed repository inside the container."""
        GIT_APPLY_CMDS = [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1 -i",
        ]
        #asyncio.run(self.runtime.run_in_session(BashAction(command=reset_command, timeout=120, check='raise')))

    def reset_swesmith_tests(self):
        """
        resets the tests in the testbed repository inside the container to their original state.
        """
        f2p_files = list(set([x.split("::", 1)[0] for x in self.ds[FAIL_TO_PASS]]))
        p2p_files = list(set([x.split("::", 1)[0] for x in self.ds[PASS_TO_PASS]]))
        all_files = list(set(f2p_files + p2p_files))
        all_files = [f for f in all_files if 
             os.path.basename(f).startswith('test_') and os.path.basename(f).endswith('.py') or
             os.path.basename(f).endswith('_test.py')]
        commit_id ='origin/main'
        reset_command = (
            f'cd "/testbed" && '
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout {commit_id} -- "{{}}" 2>/dev/null'
        )
        asyncio.run(self.runtime.run_in_session(BashAction(command=reset_command, timeout=60, check='raise')))
    # def setup_env_swesmith(self):
    #     try:
    #         commit_id = self.ds['base_commit']
    #         self.run("git fetch")
    #         self.run(f"git checkout {commit_id}")
    #         # Setup the run_test.sh script for subsequent testing.  
    #         test_command, _ = get_test_command(self.ds)
    #         eval_script_content = "\n".join(
    #             [
    #                 "#!/bin/bash",
    #                 "set -uxo pipefail",
    #                 "source /opt/miniconda3/bin/activate",
    #                 f"conda activate testbed",
    #                 f"cd testbed/",
    #                 f": '>>>>> Start Test Output'",
    #                 test_command,
    #                 f": '>>>>> End Test Output'",
    #             ]
    #         ) + "\n"
            
    #         with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
    #             temp_file.write(eval_script_content)
    #             temp_file.flush()  # Ensure content is written to disk
    #             temp_file_path = temp_file.name
            
    #         # Copy the file to container and clean up
    #         self.copy_to_container(temp_file_path, "/run_tests.sh")
    #         os.unlink(temp_file_path)  # Clean up the temporary file
            
    #         self.run("chmod +x /run_tests.sh")

    #         # Ensure can call and execute the tools in /usr/local/bin.
    #         self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
    #         self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
    #         self.run("python -m pip install chardet")
    #     except Exception as e:
    #         self.logger.error(f"Error setting up environment: {repr(e)}")
    def setup_env_swesmith(self):
        """
        sets up the environment for running tests in the testbed repository inside the container.
        """
            
        test_command, _ = get_test_command(self.ds)
        
        eval_script_content = "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"source /opt/miniconda3/bin/activate",
                f"conda activate testbed",
                f"cd /testbed",
                f": '>>>>> Start Test Output'",
                test_command,
                f": '>>>>> End Test Output'",
            ]
        ) + "\n"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
            temp_file.write(eval_script_content)
            temp_file.flush()  # Ensure content is written to disk
            temp_file_path = temp_file.name
        
        # Copy the file to container and clean up

        asyncio.run(self.runtime.upload(UploadRequest(source_path=temp_file_path,target_path="/run_tests.sh")))
        os.unlink(temp_file_path)  # Clean up the temporary file
        # self.run_command("chmod +x /run_tests.sh && python -m pip install chardet")
        
        #asyncio.run(self.runtime.run_in_session(BashAction(command='"ln -s /opt/miniconda3/envs/testbed /root/.venv"', timeout=120, check='raise')))
        asyncio.run(self.runtime.run_in_session(BashAction(command='echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc', timeout=10, check='raise')))
        asyncio.run(self.runtime.run_in_session(BashAction(command=f"chmod +x /run_tests.sh && python -m pip install chardet", timeout=60, check='raise')))
    def parse_logs(self, log_output: str) -> dict:
        """Parses the log output from the testbed repository inside the container."""
        return parse_log_fn("testbed")(log_output)
    def _calculate_reward(self):
        """Calculates the reward based on the test results from the testbed repository inside the container."""
        timeout= self._config.eval_timeout
        self.reset_swesmith_tests()
        self.setup_env_swesmith()
        
        output= asyncio.run(self.runtime.run_in_session(BashAction(command="/run_tests.sh", timeout=timeout, check='raise'))).output
        
        # print(output)
        parse = self.parse_logs(output)
        
        fail2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['FAIL_TO_PASS']]
        pass2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['PASS_TO_PASS']]
        # @(Naman, Jas): Parse the output and return the reward. This implementation is a hack rn.
        if not parse:
            return 0.0,{},{},output
        
        fail2pass_dic={}
        for test_name in fail2pass:
            if test_name not in parse:
                # Check if test_name is substring of any key
                matching_key = next((k for k in parse.keys() if test_name in k), None)

                fail2pass_dic[test_name]=matching_key
               
                
            else:
            
                fail2pass_dic[test_name]=parse[test_name]
            
        
        # Check pass2pass
        pass2pass_dic={}
        for test_name in pass2pass:
            if test_name not in parse:
                # Check if test_name is substring of any key
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                pass2pass_dic[test_name]=matching_key
           
                test_name = matching_key
            else:
                pass2pass_dic[test_name]=parse[test_name]
               
        # Check fail2pass
        for test_name in fail2pass:
            if test_name not in parse:
                # Check if test_name is substring of any key
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                if matching_key is None:
                    return 0.0,fail2pass_dic,pass2pass_dic,output
                if parse[matching_key] != 'PASSED':
                    return 0.0,fail2pass_dic,pass2pass_dic,output
                test_name = matching_key
            if parse[test_name] != 'PASSED':
                return 0.0,fail2pass_dic,pass2pass_dic,output
        
        # Check pass2pass
        for test_name in pass2pass:
            if test_name not in parse:
                # Check if test_name is substring of any key
                matching_key = next((k for k in parse.keys() if test_name in k), None)
                if matching_key is None:
                    return 0.0,fail2pass_dic,pass2pass_dic,output
                test_name = matching_key
            if parse[test_name] != 'PASSED':
                return 0.0,fail2pass_dic,pass2pass_dic,output
        return 1.0,fail2pass_dic,pass2pass_dic,output

    def _get_container_name(self) -> str:
        """Returns a unique container name based on the image name."""
        image_name_sanitized = "".join(c for c in self._config.image if c.isalnum() or c in "-_.")
        return f"{image_name_sanitized}-{uuid.uuid4()}"

    @property
    def container_name(self) -> str | None:
        return self._container_name

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
       
        if self._runtime is None:
            msg = "Runtime not started"
            raise RuntimeError(msg)
        if self._container_process is None:
            msg = "Container process not started"
            raise RuntimeError(msg)
        if self._container_process.poll() is not None:
            msg = "Container process terminated."
            output = "stdout:\n" + self._container_process.stdout.read().decode()  # type: ignore
            output += "\nstderr:\n" + self._container_process.stderr.read().decode()  # type: ignore
            msg += "\n" + output
            raise RuntimeError(msg)
        return await self._runtime.is_alive(timeout=timeout)

    async def _wait_until_alive(self, timeout: float = 10.0):
        try:
            return await _wait_until_alive(self.is_alive, timeout=timeout, function_timeout=self._runtime_timeout)
        except TimeoutError as e:
            self.logger.error("Runtime did not start within timeout. Here's the output from the container process.")
            self.logger.error(self._container_process.stdout.read().decode())  # type: ignore
            self.logger.error(self._container_process.stderr.read().decode())  # type: ignore
            assert self._container_process is not None
            await self.stop()
            raise e

    def _get_token(self) -> str:
        return str(uuid.uuid4())

    def _get_swerex_start_cmd(self, token: str) -> list[str]:
        rex_args = f"--auth-token {token}"
        pipx_install = "python3 -m pip install pipx && python3 -m pipx ensurepath"
        if self._config.python_standalone_dir:
            cmd = f"{self._config.python_standalone_dir}/python3.11/bin/{REMOTE_EXECUTABLE_NAME} {rex_args}"
        else:
            cmd = f"{REMOTE_EXECUTABLE_NAME} {rex_args} || ({pipx_install} && pipx run {PACKAGE_NAME} {rex_args})"
        # Need to wrap with /bin/sh -c to avoid having '&&' interpreted by the parent shell
        return [
            "/bin/sh",
            # "-l",
            "-c",
            cmd,
        ]

    def _pull_image(self) -> None:
        if self._config.pull == "never":
            return
        if self._config.pull == "missing" and _is_image_available(self._config.image, self._config.container_runtime):
            return
        self.logger.info(f"Pulling image {self._config.image!r}")
        self._hooks.on_custom_step("Pulling container image")
        try:
            _pull_image(self._config.image, self._config.container_runtime)
        except subprocess.CalledProcessError as e:
            msg = f"Failed to pull image {self._config.image}. "
            msg += f"Error: {e.stderr.decode()}"
            msg += f"Output: {e.output.decode()}"
            raise DockerPullError(msg) from e

    @property
    def glibc_dockerfile(self) -> str:
        # will only work with glibc-based systems
        if self._config.platform:
            platform_arg = f"--platform={self._config.platform}"
        else:
            platform_arg = ""
        return (
            "ARG BASE_IMAGE\n\n"
            # Build stage for standalone Python
            f"FROM {platform_arg} python:3.11.9-slim-bookworm AS builder\n"
            # Install build dependencies
            "RUN apt-get update && apt-get install -y \\\n"
            "    wget \\\n"
            "    gcc \\\n"
            "    make \\\n"
            "    zlib1g-dev \\\n"
            "    libssl-dev \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n\n"
            # Download and compile Python as standalone
            "WORKDIR /build\n"
            "RUN wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz \\\n"
            "    && tar xzf Python-3.11.8.tgz\n"
            "WORKDIR /build/Python-3.11.8\n"
            "RUN ./configure \\\n"
            "    --prefix=/root/python3.11 \\\n"
            "    --enable-shared \\\n"
            "    LDFLAGS='-Wl,-rpath=/root/python3.11/lib' && \\\n"
            "    make -j$(nproc) && \\\n"
            "    make install && \\\n"
            "    ldconfig\n\n"
            # Production stage
            f"FROM {platform_arg} $BASE_IMAGE\n"
            # Ensure we have the required runtime libraries
            "RUN apt-get update && apt-get install -y \\\n"
            "    libc6 \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            # Copy the standalone Python installation
            f"COPY --from=builder /root/python3.11 {self._config.python_standalone_dir}/python3.11\n"
            f"ENV LD_LIBRARY_PATH={self._config.python_standalone_dir}/python3.11/lib:${{LD_LIBRARY_PATH:-}}\n"
            # Verify installation
            f"RUN {self._config.python_standalone_dir}/python3.11/bin/python3 --version\n"
            # Install swe-rex using the standalone Python
            f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {PACKAGE_NAME}\n\n"
            f"RUN ln -s /root/python3.11/bin/{REMOTE_EXECUTABLE_NAME} /usr/local/bin/{REMOTE_EXECUTABLE_NAME}\n\n"
            f"RUN {REMOTE_EXECUTABLE_NAME} --version\n"
        )

    def _build_image(self) -> str:
        """Builds image, returns image ID."""
        self.logger.info(
            f"Building image {self._config.image} to install a standalone python to {self._config.python_standalone_dir}. "
            "This might take a while (but you only have to do it once). To skip this step, set `python_standalone_dir` to None."
        )
        dockerfile = self.glibc_dockerfile
        platform_arg = []
        if self._config.platform:
            platform_arg = ["--platform", self._config.platform]
        build_cmd = [
            self._config.container_runtime,
            "build",
            "-q",
            *platform_arg,
            "--build-arg",
            f"BASE_IMAGE={self._config.image}",
            "-",
        ]
        image_id = (
            subprocess.check_output(
                build_cmd,
                input=dockerfile.encode(),
            )
            .decode()
            .strip()
        )
        if not image_id.startswith("sha256:"):
            msg = f"Failed to build image. Image ID is not a SHA256: {image_id}"
            raise RuntimeError(msg)
        return image_id

    async def start(self):
      
        self._pull_image()
        if self._config.python_standalone_dir:
            image_id = self._build_image()
        else:
            image_id = self._config.image
        if self._config.port is None:
            self._config.port = find_free_port()
        assert self._container_name is None
        self._container_name = self._get_container_name()
        token = self._get_token()
        platform_arg = []
        if self._config.platform is not None:
            platform_arg = ["--platform", self._config.platform]
        rm_arg = []
        if self._config.remove_container:
            rm_arg = ["--rm"]
        cmds = [
            self._config.container_runtime,
            "run",
            *rm_arg,
            "-p",
            f"{self._config.port}:8000",
            *platform_arg,
            *self._config.docker_args,
            "--name",
            self._container_name,
            image_id,
            *self._get_swerex_start_cmd(token),
        ]
        cmd_str = shlex.join(cmds)
        self.logger.info(
            f"Starting container {self._container_name} with image {self._config.image} serving on port {self._config.port}"
        )
        self.logger.debug(f"Command: {cmd_str!r}")
        # shell=True required for && etc.
        self._container_process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._hooks.on_custom_step("Starting runtime")
        self.logger.info(f"Starting runtime at {self._config.port}")
        self._runtime = RemoteRuntime.from_config(
            RemoteRuntimeConfig(port=self._config.port, timeout=self._runtime_timeout, auth_token=token,host=self._config.host)
        )
        t0 = time.time()
        await self._wait_until_alive(timeout=self._config.startup_timeout)
        self.logger.info(f"Runtime started in {time.time() - t0:.2f}s")

    async def stop(self):
       
        if self._runtime is not None:
            await self._runtime.close()
            self._runtime = None

        if self._container_process is not None:
            try:
                subprocess.check_call(
                    [self._config.container_runtime, "kill", self._container_name],  # type: ignore
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.logger.warning(
                    f"Failed to kill container {self._container_name}: {e}. Will try harder.",
                    exc_info=False,
                )
            for _ in range(3):
                self._container_process.kill()
                try:
                    self._container_process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                self.logger.warning(f"Failed to kill container {self._container_name} with SIGKILL")

            self._container_process = None
            self._container_name = None

        if self._config.remove_images:
            if _is_image_available(self._config.image, self._config.container_runtime):
                self.logger.info(f"Removing image {self._config.image}")
                try:
                    _remove_image(self._config.image, self._config.container_runtime)
                except subprocess.CalledProcessError:
                    self.logger.error(f"Failed to remove image {self._config.image}", exc_info=True)

    @property
    def runtime(self) -> RemoteRuntime:
       
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime
