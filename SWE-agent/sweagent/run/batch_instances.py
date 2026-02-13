import json
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional, Union, Literal

from datasets import load_from_disk

from pydantic import BaseModel, ConfigDict, Field, model_validator
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    DummyDeploymentConfig,
    LocalDeploymentConfig,
)
from typing_extensions import Self

from sweagent.agent.problem_statement import (
    ProblemStatementConfig,
    SWEBenchMultimodalProblemStatement,
    TextProblemStatement,
)
from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig, PreExistingRepoConfig,GithubRepoRetryConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.utils.files import load_file
from sweagent.utils.log import get_logger
from swesandbox.sandbox_deployment import SandboxDeploymentConfig
from swerex.deployment.config import (
    DeploymentConfig,
    DockerDeploymentConfig,
    DummyDeploymentConfig,
    LocalDeploymentConfig,
)
logger = get_logger("swea-config", emoji="🔧")


class AbstractInstanceSource(ABC):
    """Anything that adheres to this standard can be used to load instances."""

    @abstractmethod
    def get_instance_configs(self) -> list[EnvironmentConfig]: ...
    @abstractmethod
    def get_instance_configs_ds(self,dataset) -> list[EnvironmentConfig]: ...

class BatchInstance(BaseModel):
    """A single instance in a batch of instances.
    This specifies both the environment configuration and the problem statement.
    """
    ds: Any
    env: EnvironmentConfig
    problem_statement: ProblemStatementConfig


def _slice_spec_to_slice(slice_spec: str) -> slice:
    if slice_spec == "":
        return slice(None)
    parts = slice_spec.split(":")
    values = [None if p == "" else int(p) for p in parts]
    if len(parts) == 1:
        return slice(values[0])
    if len(parts) == 2:
        return slice(values[0], values[1])
    if len(parts) == 3:
        return slice(values[0], values[1], values[2])
    msg = (
        f"Invalid slice specification: {slice_spec!r}. "
        "Here's the expected format: stop or start:stop or start:stop:step "
        "(i.e., it behaves exactly like python's list slicing `list[slice]`)."
    )
    raise ValueError(msg)


def _filter_batch_items(
    instances: list[BatchInstance], *, filter_: str, slice_: str = "", shuffle: bool = False
) -> list[BatchInstance]:
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x.problem_statement.id)
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [instance for instance in instances if re.match(filter_, instance.problem_statement.id)]
    after_filter = len(instances)
    if before_filter != after_filter:
        logger.info("Instance filter: %d -> %d instances", before_filter, after_filter)
    if slice_:
        instances = instances[_slice_spec_to_slice(slice_)]
        after_slice = len(instances)
        if before_filter != after_slice:
            logger.info("Instance slice: %d -> %d instances", before_filter, after_slice)
    return instances


class SimpleBatchInstance(BaseModel):
    """A simple way to configure a single instance in a batch of instances that all
    use similar deployment configurations.
    We added traj_id to support multiple rollouts per instance.
    """
    global_step: int = -1
    repo_type : str
    """Type of the repository. Can be 'github', 'local', or '' (empty string for pre-existing)."""
    ds : Any
    """One item from the dataset."""
    image_name: str
    problem_statement: str
    instance_id: str
    """Unique identifier of the instance. Used for logging and environment naming."""
    traj_id: str
    """Unique identifier of the trajectory. Necessary when multiple rollouts are performed per instance."""
    repo_name: str = ""
    """Specifies the repository name to use.
    For `github` repo_type, this is the git folder name.
    For `local` repo_type, this is the local path to the repo.
    For pre-existing repo_type, this is the name of the pre-existing repo. 
    """
    base_commit: str = "HEAD"
    """Used to reset repo."""
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    """Any additional data to be added to the instance.
    This data will be available when formatting prompt templates.
    """

    # Ignore instead of allow because they should be added as `extra_fields`
    model_config = ConfigDict(extra="ignore")

    def to_full_batch_instance(self, deployment: DeploymentConfig) -> BatchInstance:
        """Merge the deployment options into the `SimpleBatchInstance` object to get a full `BatchInstance`.
        We use our GithubRepoRetryConfig instead of GithubRepoConfig.
        """
        # Very important: Make a copy of the deployment config because it will be shared among instances!!!
        deployment = deployment.model_copy(deep=True)

        if "issue_images" in self.extra_fields:
            problem_statement = SWEBenchMultimodalProblemStatement(
                text=self.problem_statement,
                issue_images=self.extra_fields.pop("issue_images"),
                id=self.instance_id,
                extra_fields=self.extra_fields,
            )
        else:
            problem_statement = TextProblemStatement(
                text=self.problem_statement, id=self.traj_id, extra_fields=self.extra_fields
            )

        if not self.repo_name:
            repo = None
       
        if self.repo_type == "github":
            repo = GithubRepoRetryConfig(git_folder=self.repo_name, base_commit=self.base_commit,github_url=self.ds['repo'])
        elif self.repo_type == "local":
            repo = LocalRepoConfig(path=Path(self.repo_name), base_commit=self.base_commit)
        else:
            repo = PreExistingRepoConfig(repo_name=self.repo_name, base_commit=self.base_commit)
        if isinstance(deployment, LocalDeploymentConfig):
            if self.image_name:
                msg = "Local deployment does not support image_name"
                raise ValueError(msg)
            return BatchInstance(
                ds=self.ds,
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )
        if isinstance(deployment, DummyDeploymentConfig):
            return BatchInstance(
                ds=self.ds,
                env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
            )

        deployment.image = self.image_name  # type: ignore

        if isinstance(deployment, DockerDeploymentConfig) and deployment.python_standalone_dir is None:
            # Note: you can disable this by setting python_standalone_dir to ""
            # deployment.python_standalone_dir = "/root"  # type: ignore
            pass
        return BatchInstance(
            ds=self.ds,
            env=EnvironmentConfig(deployment=deployment, repo=repo), problem_statement=problem_statement
        )
    

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_id(cls, data):
        # Handling compatibility with swe-agent <= 1.0.1
        if isinstance(data, dict):
            if "id" in data and "instance_id" not in data:
                data["instance_id"] = data["id"]
                data.pop("id")
        return data

    # todo: Maybe populate extra fields?
    @classmethod
    def from_swe_bench(cls, instance: dict[str, Any]) -> Self:
        """Convert instances from the classical SWE-bench dataset to the `SimpleBatchInstance` format.
        
        Attributes:
            instance: A dictionary representing a single instance from the SWE-bench dataset.
        """
        iid = instance["instance_id"]
        image_name = instance.get("image_name", None)
        if image_name is None:
            # Docker doesn't allow double underscore, so we replace them with a magic token
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
            iid_image_name=f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}".lower()
            instance['image_name']=iid_image_name
        extra_fields = {}
        if "image_assets" in instance:
            issue_images = json.loads(instance["image_assets"])["problem_statement"]
            extra_fields["issue_images"] = issue_images
        
        return cls(
            repo_type=instance.get("repo_type",'github'),
            ds=instance,
            image_name=image_name,
            problem_statement=instance["problem_statement"],
            instance_id=iid,
            traj_id=instance.get("traj_id", iid),
            repo_name="testbed",
            base_commit=instance.get("base_commit",'main'),
            extra_fields=extra_fields,
        )


class InstancesFromFile(BaseModel, AbstractInstanceSource):
    """Load instances from a file."""

    path: Path
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    deployment: SandboxDeploymentConfig = Field(
        default_factory=lambda: SandboxDeploymentConfig(),
    )
    """Note that the image_name option is overwritten by the images specified in the task instances."""

    simple: Literal[True] = True
    """Convenience discriminator for (de)serialization/CLI. Do not change."""

    type: Literal["file"] = "file"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        simple_instances = [SimpleBatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class InstancesFromHuggingFace(BaseModel, AbstractInstanceSource):
    """Load instances from HuggingFace."""

    dataset_name: str
    """Name of the HuggingFace dataset. Same as when using `datasets.load_dataset`."""
    split: str = "dev"
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    deployment: SandboxDeploymentConfig = Field(
        default_factory=lambda: SandboxDeploymentConfig(),
    )
    """Deployment configuration. Note that the `image_name` option is overwritten by the images specified in the task instances.
    """
    type: Literal["huggingface"] = "huggingface"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        from datasets import load_dataset

        ds: list[dict[str, Any]] = load_dataset(self.dataset_name, split=self.split)  # type: ignore
        simple_instances: list[SimpleBatchInstance] = [SimpleBatchInstance.model_validate(instance) for instance in ds]
        instances = [instance.to_full_batch_instance(self.deployment) for instance in simple_instances]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        ds_name = "".join(l for l in self.dataset_name if l.isalnum() or l in ["-", "_"])
        return f"{ds_name}_{self.split}"


class SWEBenchInstances(BaseModel, AbstractInstanceSource):
    """Load instances from SWE-bench."""
    repo_type: str = 'github'
    """Type of repository to use. Options include 'github', 'local', and 'preexisting'."""
    database: str="/home/zeta/ais_default_code_config_repo_507090/datasets"
    """Path to the local SWE-bench database."""
    start: int =0
    """Start index of instances to load."""
    end: int =500
    """End index of instances to load."""
    subset: Literal["lite", "verified", "full", "multimodal", "multilingual"] = "lite"
    """Subset of swe-bench to use"""

    # IMPORTANT: Do not call this `path`, because then if people do not specify instance.type,
    # it might be resolved to ExpertInstancesFromFile or something like that.
    # path_override: str | Path | None = None
    # """Allow to specify a different huggingface dataset name or path to a huggingface
    # dataset. This will override the automatic path set by `subset`.
    # """
    model_patch_file: str =None
    """Path to a json file containing model patches for the instances. Used for 
    MiniSandbox evaluation with patches."""

    split: str='test'

    deployment: DeploymentConfig = Field(
        default_factory=lambda: SandboxDeploymentConfig(),
    )
   

    type: Literal["swe_bench"] = "swe_bench"
    """Discriminator for (de)serialization/CLI. Do not change."""

    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    evaluate: bool = False
    """Run sb-cli to evaluate"""

    def _get_dataset_path(self) -> str:
        # if self.path_override is not None:
        #     return str(self.path_override)
        # dataset_mapping = {
        #     "full": "princeton-nlp/SWE-Bench",
        #     "verified": f"/home/zeta/ais_default_code_config_repo_507090/datasets/SWE-bench_Verified/data",
        #     "lite": "princeton-nlp/SWE-Bench_Lite",
        #     "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
        #     "multilingual": "swe-bench/SWE-Bench_Multilingual",
        # }

        # if self.subset not in dataset_mapping:
        #     msg = f"Unsupported subset: {self.subset}"
        #     raise ValueError(msg)

        return self.database
    def get_instance_configs_ds(self,dataset) -> list[BatchInstance]:
        pass
    def get_instance_configs(self) -> list[BatchInstance]:
        """Load instances from SWE-bench dataset. This function is used for run_batch.py"""
        from datasets import load_dataset

        ds: list[dict[str, Any]] = load_dataset(self._get_dataset_path(), split=self.split).shuffle(42).select(range(self.start,self.end))  # type: ignore
        
        
        """
        test patches is json like below:
            {
        "sympy__sympy-19954": {
            "reward": null,
            "test_out": null,
            "p2p": null,
            "f2p": null,
            "model_name_or_path": "out-qwen2.5-3Bcoder-docker5k",
            "instance_id": "sympy__sympy-19954",
            "model_patch": "diff --git a/reproduce_error.py b/reproduce_error.py\nnew file mode 100644\nindex 0000000000..52decb4042\n--- /dev/null\n+++ b/reproduce_error.py\n@@ -0,0 +1,11 @@\n+from sympy.combinatorics import DihedralGroup, Permutation\n+\n+G = DihedralGroup(18)\n+\n+S2 = G.sylow_subgroup(p=2)\n+print(\"S2 order:\", S2.order())\n+\n+# Try with a larger group\n+G2 = DihedralGroup(2*25)\n+S2_large = G2.sylow_subgroup(p=2)\n+print(\"S2_large order:\", S2_large.order())\n\\ No newline at end of file\ndiff --git a/sympy/combinatorics/perm_groups.py b/sympy/combinatorics/perm_groups.py\nindex de94ddabb4..ddc9c06ad5 100644\n--- a/sympy/combinatorics/perm_groups.py\n+++ b/sympy/combinatorics/perm_groups.py\n@@ -2200,6 +2200,9 @@ def _number_blocks(blocks):\n                         # i-th block system is not minimal\n                         del num_blocks[i], blocks[i]\n                         to_remove.append(rep_blocks[i])\n+                        # After deletion, we need to adjust the indices\n+                        # because we removed elements from both lists\n+                        break\n                     elif len(r) < len(rep) and r.issubset(rep):\n                         # the system being checked is not minimal\n                         minimal = False\n"
        },
        "django__django-13821": {
            "reward": null,
            "test_out": null,
            "p2p": null,
            "f2p": null,
            "model_name_or_path": "out-qwen2.5-3Bcoder-docker5k",
            "instance_id": "django__django-13821",
            "model_patch": ""
        },
        """
        
        import json
        if self.model_patch_file is not None:
            new_ds=[]
            with open(self.model_patch_file,'r') as f:
                test_patches = json.load(f)
            for i in range(len(ds)):
                instance_id = ds[i]['instance_id']
                d={}
                for key in ds[i]:
                    d[key]=ds[i][key]

                if instance_id in test_patches:
                    d['patch'] = test_patches[instance_id]['model_patch']
                else:
                    d['patch'] = ''
                new_ds.append(d)
            ds=new_ds

        instances = [
            SimpleBatchInstance.from_swe_bench({**instance, 'repo_type': self.repo_type}).to_full_batch_instance(self.deployment) for instance in ds
        ]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)
    
    @property
    def id(self) -> str:
        return f"swe_bench_{self.subset}_{self.split}"


class ExpertInstancesFromFile(BaseModel, AbstractInstanceSource):
    """Load instances from a file. The difference to `InstancesFromFile` is that the instances are configured as full
    `EnvironmentInstanceConfig` objects, i.e., we could specify separate deployment configurations etc.
    """

    path: Path
    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    type: Literal["expert_file"] = "expert_file"
    """Discriminator for (de)serialization/CLI. Do not change."""

    def get_instance_configs(self) -> list[BatchInstance]:
        instance_dicts = load_file(self.path)
        instances = [BatchInstance.model_validate(instance_dict) for instance_dict in instance_dicts]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return self.path.stem


class SWESmithInstances(BaseModel, AbstractInstanceSource):
    """Load instances from SWE-smith."""
    

    repo_type: str = 'github'
    """Type of the repository. Can be 'github', 'local', or anything else for pre-existing repositories.
    If the repo_type is 'github', we will use git fetch to get the code from the specified repo and base_commit in the instance config.
    If the repo_type is 'local', we do not support yet
    If the repo_type is anything else (e.g., 'preexisting'), we will assume that the code is already present in the deployment and do not perform any operations. The instance config should still specify the repo_name and base_commit for resetting the repo to the correct state.
    """
    load_from_disk: bool=False
    """Whether to load the dataset from local disk using `load_from_disk`."""
    num_rollouts_per_instance: int=-1
    """Number of rollouts to perform per instance. If >1, the instances will be duplicated accordingly."""
    start: int=0
    """Start index of instances to load."""
    end: int=500
    """End index of instances to load."""
    path: str
    """Name or path of the dataset."""
    split: str
    """Split of the dataset to load."""
    deployment: DeploymentConfig = Field(
        default_factory=lambda: SandboxDeploymentConfig(),
    )


    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    type: Literal["swesmith"] = "swesmith"
    """Discriminator for (de)serialization/CLI. Do not change."""
    def get_instance_configs_ds(self,dataset) -> list[BatchInstance]:
        pass
    def get_instance_configs(self) -> list[BatchInstance]:
        """Load instances from SWE-smith dataset. This function is used for run_batch.py
        It duplicates instances if num_rollouts_per_instance>1.
        """
        num_rollouts_per_instance = self.num_rollouts_per_instance
        def convert_instance_dict(instance_dict) -> dict[str, Any]:
            instance_dict['repo_type'] = self.repo_type
            instance_dict["id"] = instance_dict["instance_id"]
            # todo: The base_commit is currently incorrect
            instance_dict["traj_id"] = instance_dict.get("traj_id", instance_dict["instance_id"])
            instance_dict["base_commit"] = instance_dict["id"] if "base_commit" not in instance_dict else instance_dict["base_commit"]
            instance_dict["problem_statement"] = instance_dict.get("problem_statement", "")
            instance_dict["repo_name"] = "testbed"
            instance_dict["extra_fields"] = {"fail_to_pass": instance_dict["FAIL_TO_PASS"]}
            instance_dict['ds']=instance_dict
            return instance_dict
        from datasets import load_dataset,load_from_disk
        
        if self.load_from_disk:
            if self.start<self.end:
                instance_dicts=load_from_disk(self.path).shuffle(42)
                instance_dicts=instance_dicts.select(range(self.start,self.end))
            else:
                instance_dicts=load_from_disk(self.path).shuffle(42)
                data_len = len(instance_dicts)
                instance_dicts=instance_dicts.select(range(self.start,data_len))
        else:
            if self.start<self.end:
                instance_dicts= load_dataset(self.path, split=self.split).shuffle(42)
                instance_dicts=instance_dicts.select(range(self.start,self.end))
            else:
                instance_dicts= load_dataset(self.path, split=self.split).shuffle(42)
                data_len = len(instance_dicts)
                instance_dicts=instance_dicts.select(range(self.start,data_len))
        
        if self.num_rollouts_per_instance>1:
            new_instance_dicts = []
            import copy
            import tqdm
            for instance in tqdm.tqdm(instance_dicts, desc="Duplicating instances"):
                for i in range(self.num_rollouts_per_instance):
                    new_instance = copy.deepcopy(instance)
                    new_instance["base_commit"] = instance['instance_id'] if 'base_commit' not in instance else instance['base_commit']
                    new_instance['traj_id'] = f"{instance['instance_id']}_{i}"
                    
                    new_instance_dicts.append(new_instance)
            instance_dicts=new_instance_dicts

        instances = [
            SimpleBatchInstance.model_validate(convert_instance_dict(instance_dict)).to_full_batch_instance(
                self.deployment
            )
            for instance_dict in instance_dicts
        ]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)

    @property
    def id(self) -> str:
        return "swesmith"



class SkyRLInstances(BaseModel, AbstractInstanceSource):
    """Load instances from RL data (smith format)."""
    
    # dataset: Optional[List[Dict[str, Any]]]
    repo_type: str = 'github'
    """Type of the repository. Can be 'github', 'local', or '' (empty string for pre-existing)."""
    path: str
    """Name or path of the dataset."""
    split: str
    """Split of the dataset to load."""
    deployment: DeploymentConfig = Field(
        default_factory=lambda: SandboxDeploymentConfig(),
    )
    

    filter: str = ".*"
    """Regular expression to filter the instances by instance id."""
    slice: str = ""
    """Select only a slice of the instances (after filtering by `filter`).
    Possible values are stop or start:stop or start:stop:step.
    (i.e., it behaves exactly like python's list slicing `list[slice]`).
    """
    shuffle: bool = False
    """Shuffle the instances (before filtering and slicing)."""

    type: Literal["skyrl"] = "skyrl"
    """Discriminator for (de)serialization/CLI. Do not change."""
    
    def get_instance_configs(self) -> list[BatchInstance]:
       
        def convert_instance_dict(instance_dict) -> dict[str, Any]:
            instance_dict['repo_type'] = self.repo_type
            instance_dict["id"] = instance_dict["instance_id"]
            # todo: The base_commit is currently incorrect
            instance_dict["traj_id"] = instance_dict.get("traj_id", instance_dict["instance_id"])
            instance_dict["base_commit"] = instance_dict["id"] if "base_commit" not in instance_dict else instance_dict["base_commit"]
            instance_dict["problem_statement"] = instance_dict.get("problem_statement", "")
            instance_dict["repo_name"] = "testbed"
            instance_dict["extra_fields"] = {"fail_to_pass": instance_dict["FAIL_TO_PASS"]}
            instance_dict['ds']=instance_dict
            return instance_dict
        # asset self has dataset
        assert hasattr(self,'dataset'),"SkyRLInstances requires dataset attribute"
        instance_dicts = self.dataset 
        
        instances = [
            SimpleBatchInstance.model_validate(convert_instance_dict(instance_dict)).to_full_batch_instance(
                self.deployment
            )
            for instance_dict in instance_dicts
        ]
        
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)
    

    def get_instance_configs_ds(self,dataset) -> list[BatchInstance]:
        """Load instances from RL data (smith format). This function is used for SkyRL"""
        
        def convert_instance_dict(instance_dict) -> dict[str, Any]:
            instance_dict['repo_type'] = self.repo_type
            instance_dict["id"] = instance_dict["instance_id"]
            # todo: The base_commit is currently incorrect
            instance_dict["traj_id"] = instance_dict.get("traj_id", instance_dict["instance_id"])
            instance_dict["base_commit"] = instance_dict["id"] if "base_commit" not in instance_dict else instance_dict["base_commit"]
            instance_dict["problem_statement"] = instance_dict.get("problem_statement", "")
            instance_dict["repo_name"] = "testbed"
            instance_dict["extra_fields"] = {"fail_to_pass": instance_dict["FAIL_TO_PASS"]}
            instance_dict['ds']=instance_dict
            return instance_dict
        
        instance_dicts = dataset

        instances = [
            SimpleBatchInstance.model_validate(convert_instance_dict(instance_dict)).to_full_batch_instance(
                self.deployment
            )
            for instance_dict in instance_dicts
        ]
        return _filter_batch_items(instances, filter_=self.filter, slice_=self.slice, shuffle=self.shuffle)
    
    @property
    def id(self) -> str:
        return "skyrl"


BatchInstanceSourceConfig = (
    InstancesFromHuggingFace | InstancesFromFile | SWEBenchInstances | ExpertInstancesFromFile | SWESmithInstances | SkyRLInstances
)
