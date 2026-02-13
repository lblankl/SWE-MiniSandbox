import asyncio
import os
from pathlib import Path
from typing import Any, Literal, Protocol

from git import InvalidGitRepositoryError
from git import Repo as GitRepo
from pydantic import BaseModel, ConfigDict, Field
from swerex.deployment.abstract import AbstractDeployment
from swerex.runtime.abstract import Command, UploadRequest,BashAction
from typing_extensions import Self

from sweagent.utils.github import _parse_gh_repo_url
from sweagent.utils.log import get_logger

logger = get_logger("swea-config", emoji="🔧")

from swesandbox.utils import copytree_via_tar,tar_extract

class Repo(Protocol):
    """Protocol for repository configurations."""
    git_folder: str
    base_commit: str
    repo_name: str

    def copy_repo(self, deployment: AbstractDeployment): ...

    def copy2(self,deployment,**kwargs: Any): ...

    def get_reset_commands(self) -> list[str]: ...


def _get_git_reset_commands(base_commit: str) -> list[str]:
    return [
        "git fetch",
        "git status",
        "git restore .",
        "git reset --hard",
        f"git checkout {base_commit}",
        "git clean -fdq",
    ]


class PreExistingRepoConfig(BaseModel):
    """Use this to specify a repository that already exists on the deployment.
    This is important because we need to cd to the repo before running the agent.

    Note: The repository must be at the root of the deployment.
    """
    repo_name: str
    """The repo name (the repository must be located at the root of the deployment)."""
    base_commit: str = Field(default="HEAD")
    """The commit to reset the repository to. The default is HEAD,
    i.e., the latest commit. You can also set this to a branch name (e.g., `dev`),
    a tag (e.g., `v0.1.0`), or a commit hash (e.g., `a4464baca1f`).
    SWE-agent will then start from this commit when trying to solve the problem.
    """

    type: Literal["preexisting"] = "preexisting"
    """Discriminator for (de)serialization/CLI. Do not change."""

    reset: bool = True
    """If True, reset the repository to the base commit after the copy operation."""

    model_config = ConfigDict(extra="forbid")

    def copy_repo(self, deployment: AbstractDeployment):
        """Does nothing."""
        pass
    def copy2(self,deployment,**kwargs: Any):
        pass
    def get_reset_commands(self) -> list[str]:
        """Issued after the copy operation or when the environment is reset."""
        if self.reset:
            return _get_git_reset_commands(self.base_commit)
        return []


class LocalRepoConfig(BaseModel):
    path: Path
    base_commit: str = Field(default="HEAD")
    """The commit to reset the repository to. The default is HEAD,
    i.e., the latest commit. You can also set this to a branch name (e.g., `dev`),
    a tag (e.g., `v0.1.0`), or a commit hash (e.g., `a4464baca1f`).
    SWE-agent will then start from this commit when trying to solve the problem.
    """
    git_folder: str
    type: Literal["local"] = "local"
    """Discriminator for (de)serialization/CLI. Do not change."""

    model_config = ConfigDict(extra="forbid")
    def copy2(self,deployment,**kwargs: Any):
        pass
    @property
    def repo_name(self) -> str:
        """Set automatically based on the repository name. Cannot be set."""
        return Path(self.path).resolve().name.replace(" ", "-").replace("'", "")

    # Let's not make this a model validator, because it leads to cryptic errors.
    # Let's just check during copy instead.
    def check_valid_repo(self) -> Self:
        try:
            repo = GitRepo(self.path, search_parent_directories=True)
        except InvalidGitRepositoryError as e:
            msg = f"Could not find git repository at {self.path=}."
            raise ValueError(msg) from e
        if repo.is_dirty() and "PYTEST_CURRENT_TEST" not in os.environ:
            msg = f"Local git repository {self.path} is dirty. Please commit or stash changes."
            raise ValueError(msg)
        return self

    def copy_repo(self, deployment: AbstractDeployment):
        self.check_valid_repo()
        asyncio.run(
            deployment.runtime.upload(UploadRequest(source_path=str(self.path), target_path=f"/{self.repo_name}"))
        )
        r = asyncio.run(
            deployment.runtime.execute(Command(command=f"chown -R root:root /{self.repo_name}", shell=True))
        )
        if r.exit_code != 0:
            msg = f"Failed to change permissions on copied repository (exit code: {r.exit_code}, stdout: {r.stdout}, stderr: {r.stderr})"
            raise RuntimeError(msg)

    def get_reset_commands(self) -> list[str]:
        """Issued after the copy operation or when the environment is reset."""
        return _get_git_reset_commands(self.base_commit)


class GithubRepoConfig(BaseModel):
    github_url: str
    git_folder: str
    base_commit: str = Field(default="HEAD")
    """The commit to reset the repository to. The default is HEAD,
    i.e., the latest commit. You can also set this to a branch name (e.g., `dev`),
    a tag (e.g., `v0.1.0`), or a commit hash (e.g., `a4464baca1f`).
    SWE-agent will then start from this commit when trying to solve the problem.
    """

    clone_timeout: float = 500
    """Timeout for git clone operation."""

    type: Literal["github"] = "github"
    """Discriminator for (de)serialization/CLI. Do not change."""

    model_config = ConfigDict(extra="forbid")
    def copy2(self,deployment,**kwargs: Any):
        pass
    def model_post_init(self, __context: Any) -> None:
        if self.github_url.count("/") == 1:
            self.github_url = f"https://github.com/{self.github_url}"

    @property
    def repo_name(self) -> str:
        org, repo = _parse_gh_repo_url(self.github_url)
        return f"{org}__{repo}"

    def _get_url_with_token(self, token: str) -> str:
        """Prepend github token to URL"""
        if not token:
            return self.github_url
        if "@" in self.github_url:
            logger.warning("Cannot prepend token to URL. '@' found in URL")
            return self.github_url
        _, _, url_no_protocol = self.github_url.partition("://")
        return f"https://{token}@{url_no_protocol}"

    def copy_repo(self, deployment: AbstractDeployment):
        """Clones the repository to the sandbox."""
        base_commit = self.base_commit
        github_token = os.getenv("GITHUB_TOKEN", "")
        url = self._get_url_with_token(github_token)
        asyncio.run(
            deployment.runtime.execute(
                Command(
                    command=" && ".join(
                        (
                            f"mkdir /{self.repo_name}",
                            f"cd /{self.repo_name}",
                            "git init",
                            f"git remote add origin {url}",
                            f"git fetch --depth 1 origin {base_commit}",
                            "git checkout FETCH_HEAD",
                            "cd ..",
                        )
                    ),
                    timeout=self.clone_timeout,
                    shell=True,
                    check=True,
                )
            ),
        )

    def get_reset_commands(self) -> list[str]:
        """Issued after the copy operation or when the environment is reset."""
        return _get_git_reset_commands(self.base_commit)


class GithubRepoRetryConfig(BaseModel):
    """Configuration for cloning a GitHub repository with retry logic."""
    git_folder: str
    """The folder name where the repository will be cloned."""

    github_url: str
    """The URL of the GitHub repository to clone."""
    base_commit: str = Field(default="HEAD")
    """The commit to reset the repository to. The default is HEAD,
    i.e., the latest commit. You can also set this to a branch name (e.g., `dev`),
    a tag (e.g., `v0.1.0`), or a commit hash (e.g., `a4464baca1f`).
    SWE-agent will then start from this commit when trying to solve the problem.
    """

    clone_timeout: float = 60
    """Timeout for git clone operation."""

    type: Literal["github"] = "github"
    """Discriminator for (de)serialization/CLI. Do not change."""

    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: Any) -> None:
        if self.github_url.count("/") == 1:
            self.github_url = f"https://github.com/{self.github_url}"

    @property
    def repo_name(self) -> str:
        """Set automatically based on the repository name. Cannot be set."""
        return self.git_folder

    def _get_url_with_token(self, token: str) -> str:
        """Prepend github token to URL"""
        if not token:
            return self.github_url
        if "@" in self.github_url:
            logger.warning("Cannot prepend token to URL. '@' found in URL")
            return self.github_url
        _, _, url_no_protocol = self.github_url.partition("://")
        return f"https://{token}@{url_no_protocol}"

    def copy_repo(self, deployment: AbstractDeployment):
        """Clones the repository to the sandbox."""
        base_commit = self.base_commit
        github_token = os.getenv("GITHUB_TOKEN", "")
        url = self._get_url_with_token(github_token)
        asyncio.run(
            deployment.runtime.execute(
                Command(
                    command=" && ".join(
                        (
                            f"mkdir /{self.repo_name}",
                            f"cd /{self.repo_name}",
                            "git init",
                            f"git remote add origin {url}",
                            f"git fetch --depth 1 origin {base_commit}",
                            "git checkout FETCH_HEAD",
                            "cd ..",
                        )
                    ),
                    timeout=self.clone_timeout,
                    shell=True,
                    check=True,
                )
            ),
        )

    def copy2(self, deployment: AbstractDeployment,try_count=3,local_path=None):
        """This function copies the repository to the deployment.
        It first checks if a local path is provided and exists. If so, it extracts
        the repository from the local path to the deployment. If not, it clones
        the repository from GitHub with retry logic.
        For repo matplotlib/matplotlib, it also extracts freetype and qhull tarballs to /testbed/build.
        This is due to the network limitations in our experimental environment, its original implementation is to directly download from the internet.

        Attributes:
            deployment (AbstractDeployment): The deployment to copy the repository to.
            try_count (int): The number of retry attempts for cloning the repository.
            local_path (str|None): The local path to the repository archive. If provided and exists, the repository will be extracted from this path instead of cloning from GitHub.

        """
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        zip_dir = os.path.abspath(os.path.join(current_dir, "../../../zip"))
        
        # if local_path exist
        if deployment._config.force_rebuild:
            if local_path is not None and os.path.exists(local_path):
                os.remove(local_path)
        if local_path is not None and os.path.exists(local_path):
            #directly copy to deployment.root_dir/deployment.git_folder
            
       
            tar_extract(local_path,os.path.join(deployment.root_dir,self.git_folder),threads=2)
            if deployment.ds['repo']=='matplotlib/matplotlib':
                # cp /home/zeta/SWE/SWE/zip/freetype-2.6.1.tar.gz to sandbox /testbed/build
               
                deployment.extract_freetype_tarball(f"{zip_dir}/freetype-2.6.1.tar.gz",'/testbed/build')
                deployment.extract_freetype_tarball(f"{zip_dir}/qhull-2020-src-8.0.2.tgz",'/testbed/build')
            return True

        
        base_commit = self.base_commit
        github_token = os.getenv("GITHUB_TOKEN", "")
        url = self._get_url_with_token(github_token)
   
        asyncio.run(
                deployment.runtime.run_in_session(
                    BashAction(
                        command=" && ".join(
                            (
                                
                                'cd /',
                                f"mkdir /{self.git_folder}" ,
                                f"cd /{self.git_folder}",
                                "git init",
                                f"git remote add origin {url}"
                            )
                        ),
                        timeout=self.clone_timeout,
                        check='raise',
                    )
                ),
        )
        #impelement retry logic
        count=0
        while count<try_count:
            try:
                asyncio.run(
                    deployment.runtime.run_in_session(
                        BashAction(
                            command=" && ".join(
                                (
                                   
                                    f"git fetch --depth 1 origin {base_commit}",
                                    "git checkout FETCH_HEAD",
                                    "cd ..",
                                )
                            ),
                            timeout=self.clone_timeout,
                            check='raise',
                        )
                    ),
                )
                break
            except Exception as e:
                print("Retry copy repo")
                count+=1
                if count==try_count:
                    raise e
                logger.warning(f"Git clone failed, retrying {count}/{try_count}...")
        if deployment.ds['repo']=='matplotlib/matplotlib':
            # cp /home/zeta/SWE/SWE/zip/freetype-2.6.1.tar.gz to sandbox /testbed/build
            deployment.extract_freetype_tarball(f"{zip_dir}/freetype-2.6.1.tar.gz",'/testbed/build')
            deployment.extract_freetype_tarball(f"{zip_dir}/qhull-2020-src-8.0.2.tgz",'/testbed/build')
        return False
    def get_reset_commands(self) -> list[str]:
        
        return _get_git_reset_commands(self.base_commit)


RepoConfig = LocalRepoConfig | GithubRepoConfig | PreExistingRepoConfig | GithubRepoRetryConfig


# def repo_from_simplified_input(
#     *, input: str, base_commit: str = "HEAD", type: Literal["local", "github", "preexisting", "auto"] = "auto"
# ) -> RepoConfig:
#     """Get repo config from a simplified input.

#     Args:
#         input: Local path or GitHub URL
#         type: The type of repo. Set to "auto" to automatically detect the type
#             (does not work for preexisting repos).
#     """
#     if type == "local":
#         return LocalRepoConfig(path=Path(input), base_commit=base_commit)
#     if type == "github":
#         return GithubRepoConfig(github_url=input, base_commit=base_commit)
#     if type == "preexisting":
#         return PreExistingRepoConfig(repo_name=input, base_commit=base_commit)
#     if type == "auto":
#         if input.startswith("https://github.com/"):
#             return GithubRepoConfig(github_url=input, base_commit=base_commit)
#         else:
#             return LocalRepoConfig(path=Path(input), base_commit=base_commit)
#     msg = f"Unknown repo type: {type}"
#     raise ValueError(msg)

