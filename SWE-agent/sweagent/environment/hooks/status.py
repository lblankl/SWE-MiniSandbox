from collections.abc import Callable

from sweagent.environment.hooks.abstract import EnvHook
from sweagent.environment.repo import Repo, RepoConfig


class SetStatusEnvironmentHook(EnvHook):
    def __init__(self, id: str, callable: Callable[[str, str], None]):
        self._callable = callable
        self._id = id

    def _update(self, message: str):
        self._callable(self._id, message)

    def on_copy_repo_started(self, repo: RepoConfig | Repo):
        self._update(f"Copying repo {repo.repo_name}")

    def on_start_deployment(self):
        self._update("Starting deployment")

    def on_install_env_started(self):
        self._update("Installing environment")

    def on_environment_startup(self):
        self._update("Starting environment")

    def on_close(self):
        self._update("Closing environment")
    
    def on_reset(self):
        self._update("reset repo")

    def on_post_init(self):
        self._update("Post initialization")
    
    def on_pre_check_finish(self):
        self._update("Pre check finish")
    
    def on_getting_testspec(self):
        self._update("Getting testspec")

    def on_running_Pathcmds(self):
        self._update("Running Pathcmds")
    def on_getting_env(self):
        self._update("Getting environment script")
    def on_getting_eval(self):
        self._update("Getting evaluation script")

    def on_copying_shared_venv(self):
        self._update("Copying shared virtual environment")


