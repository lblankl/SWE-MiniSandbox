import logging
from typing import Any, Literal
import venv
from typing_extensions import Self
from pydantic import BaseModel, ConfigDict, Field, model_validator
import asyncio
import os
import uuid
import copy
from pathlib import Path
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.config import LocalDeploymentConfig
from swerex.deployment.hooks.abstract import CombinedDeploymentHook, DeploymentHook
from swerex.exceptions import DeploymentNotStartedError
from swerex.runtime.abstract import IsAliveResponse
from swerex.runtime.sandbox import LocalRuntime
from swerex.utils.log import get_logger
from r2egym.repo_analysis.execution_log_parser import parse_log_fn
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
import os
import shutil
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from .swebench_utils.test_spec import get_test_specs_from_dataset,get_test_specs_from_ds
from swerex.runtime.abstract import BashAction
from r2egym.swesmith.utils import get_test_command,get_install_commands
from .swe_bench_instance_map import instance_map,instance_to_skip
import time
import contextlib
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
from r2egym.swesmith.constants import (
    KEY_IMAGE_NAME,
    MAP_REPO_TO_SPECS,
    get_spec_smith
)
from r2egym.swesmith.utils import get_repo_commit_from_image_name
import os
import tarfile
import tempfile
from swesandbox.utils import copytree_via_tar, tar_extract
from swesandbox.customer_instance import custom_install_cmd,custom_test_cmd
def get_install_commands_wrapper(ds):
    instance_id=ds.get('instance_id','default')
    custom_install = custom_install_cmd(instance_id)
    if custom_install is not None:
        return custom_install
    else:
        return get_install_commands(ds)
def get_test_commands_wrapper(ds):
    instance_id=ds.get('instance_id','default')
    custom_test = custom_test_cmd(instance_id)
    if custom_test is not None:
        return custom_test
    else:
        return get_test_command(ds)  

__all__ = ["SandboxDeployment", "SandboxDeploymentConfig"]
def get_deployment_from_config(config,ds,bundles):
    if not isinstance(config,SandboxDeploymentConfig):
        raise ValueError("config must be an instance of SandboxDeploymentConfig")
        return get_deployment(config.deployment)
    config.ds=ds
    config.bundles=bundles
    current_config=copy.deepcopy(config)
    data_type=current_config.data_type
    
    git_id=map_to_git_id(ds,data_type=data_type)
    # instance_id=ds.get("instance_id",git_id)
    traj_id=ds.get("traj_id", ds.get("instance_id",git_id))
    current_config.root_dir=create_unique_folder(current_config.root_base,traj_id)
    
    
    return current_config.get_deployment()
def create_unique_folder(base_dir,traj_id):
    while True:
        folder_name = traj_id+'_'+str(uuid.uuid4())  # 全局唯一 ID
        folder_path = os.path.join(base_dir, folder_name)
        try:
            os.makedirs(folder_path,exist_ok=True)  # 原子操作
            return folder_path  # 成功创建，直接返回
        except FileExistsError:
            # 冲突，重试
            continue
   

def map_to_git_id(ds,data_type):
  
    if data_type=="swebench":
        return ds["repo"]
    elif data_type=="swesmith":
        if 'repo' in ds:
            return ds['repo']
        instance_id=ds['instance_id']
        git_id='.'.join(instance_id.split('.')[:2])
        return git_id
    else:
        return ds["repo"]

class SandboxDeploymentConfig(BaseModel):
    """
    The deployment config for MinimalSandboxDeployment.
    
    Attributes:
        force_rebuild: Whether to force rebuild the sandbox environment.
        eval_timeout: Timeout for evaluation in seconds.
        cmd_list: List of additional commands to run during environment setup.
        cache_git: Whether to cache the git repository.
        delete_after_create: Whether to delete the virtual environment and git cache after creation.
        git_base_path: Base path for git repositories cache
        data_type: Type of data, either "swebench" or "swesmith".
        ds: Dataset information, an item of dataset list.
        root_base: Base directory for sandboxes.
        root_dir: Root directory for the specific sandbox instance.
        git_folder: Folder name for the git repository within the sandbox.
        tool_path: Path to tools within the sandbox.
        bundles: List of tool bundles to include in the sandbox.
        needed_packages: List of additional packages needed in the environment.
        conda_env: Path to the conda environment.
        shared_venv: Path to the shared virtual environment directory.
        abs_tool_path: Absolute path to tools within the sandbox.
        type: Discriminator for (de)serialization/CLI, fixed as "sandbox".
    """
    
    force_rebuild: bool =False
    eval_timeout: int = 300
    cached: bool = False
    cmd_list: list = []
    cache_git: bool = True
    delete_after_create: bool = False # for smith filter, we want to delte the venv and git cache to reduce the storage consumption
    git_base_path: str ="/sandbox/git"
    data_type: str="swebench"
    ds : Any =None
    root_base: str="/sandbox"
    root_dir: Any=None
    git_folder: str = "testbed"
    tool_path: str = "/tools"
    bundles: list = []
    needed_packages: list = []
    conda_env: str = "/miniconda3"
    shared_venv: str = "/SWE/shared"
    abs_tool_path: str = "/tools" 
    type: Literal["sandbox"] = "sandbox"
    """Discriminator for (de)serialization/CLI. Do not change."""

    model_config = ConfigDict(extra="allow")
    
    

    def get_deployment(self) -> AbstractDeployment:
        """Creates a `SandboxDeployment` from this config."""
        assert self.ds is not None, "ds must be set in SandboxDeploymentConfig"
        assert len(self.bundles) > 0, "bundles must be set in SandboxDeploymentConfig"
        assert self.root_dir is not None, "root_dir must be set in SandboxDeploymentConfig"
        return SandboxDeployment.from_config(self)
    def __repr__(self):
        return (f"SandboxConfig(root_dir={self.root_dir}, "
                f"git_folder={self.git_folder}, tool_path={self.tool_path}, "
                f"bundles={self.bundles}, needed_packages={self.needed_packages}, "
                f"conda_env={self.conda_env}, shared_venv={self.shared_venv})")
    def __init__(self,**data):
        super().__init__(**data)
        os.makedirs(self.root_base, exist_ok=True)
        os.makedirs(self.shared_venv, exist_ok=True)
        #check the folder if exist, del and re create
        # if os.path.exists(self.root_base):
        #     shutil.rmtree(self.root_base)
        
        
class SandboxDeployment(AbstractDeployment):
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        **kwargs: Any,
    ):
        """
        Initializes the MiniSandboxDeployment.

        Attributes:
            logger: Logger for the deployment.
            **kwargs: Additional keyword arguments for the deployment config.
        """
        self._runtime = None
        self.logger = logger or get_logger("rex-deploy")
        self._config = SandboxDeploymentConfig(**kwargs)
        self._hooks = CombinedDeploymentHook()
        self.__dict__.update(kwargs)
        self.git_id =  map_to_git_id(self._config.ds,self._config.data_type)
        self.version = self._config.ds.get("version", "latest")
        cached_git_name=self.git_id + '/' + self.version + '/' + self.ds.get('instance_id','default')
       
        self.cached_git_dir=os.path.join(self.git_base_path,cached_git_name,'testbed.tar.gz')
        #create sandbox root dir
        if not os.path.exists(self._config.root_dir):
            os.makedirs(self._config.root_dir, exist_ok=True)
        if not os.path.exists(self._config.git_base_path):
            os.makedirs(self._config.git_base_path, exist_ok=True)
        
        tg_tool_path=os.path.join(self._config.root_dir,self._config.abs_tool_path.lstrip('/'))
        
        copytree_via_tar(self.tool_path,tg_tool_path,dirs_exist_ok=True)
        if self._config.data_type=="swebench":
            repo=self._config.ds['repo']
            version=self.version
            self.py_version = MAP_REPO_VERSION_TO_SPECS[repo][version].get("python", 3.10)
            
        else:

            repo, commit = get_repo_commit_from_image_name(self._config.ds[KEY_IMAGE_NAME])
            specs = get_spec_smith(repo,commit)
            self.py_version = specs.get("python", 3.10)
    def startup(self):
        """Generates the command to start the sandboxed environment."""
        return self.startup_old()
   
    def startup_new(self):
        base_root=self._config.BASE_ROOT
        # We first mount all the necessary folders to base_root
        # then we rbind mount base_root to self._config.root_dir

        mount_shared_cmds=[
            #mount --rbind /home/base_root /home/sandbox1
            "mount --rbind {} {}".format(base_root,self._config.root_dir)
        ]
        self._ps1='SHELLPS1PREFIX'
        initfile_path = os.path.join(self._config.root_dir, "init_ps1.sh")
        with open(initfile_path, "w") as f:
            f.write(f"export USER=${{USER:-$(whoami)}}\nexport PS1='{self._ps1}'\n")

        chroot_cmd = f"exec chroot {self._config.root_dir} /bin/bash --noprofile --norc --init-file {initfile_path}"
        full_cmd = ' && '.join(mount_shared_cmds + [chroot_cmd])
        mount_ns_cmd = f"unshare --mount sh -c \"{full_cmd}\""
        full_cmd=mount_ns_cmd
        #write this command to a shell script under root_dir/start_sandbox.sh
        script_path = os.path.join(self._config.root_dir, "start_sandbox.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n")
            f.write(full_cmd + "\n")
        os.chmod(script_path, 0o755)
        run_shell_cmd=f"{script_path}"
        return run_shell_cmd
    def startup_old(self):
        #mount namespace cmd
        
        
        self.mount_points = [
            # 普通命令和库，只读
            ("/bin", "ro","/bin"),
            ("/sbin", "ro","/sbin"),
            ("/lib", "ro","/lib"),
            ("/lib64", "ro","/lib64"),
            ("/usr", "ro","/usr"),
            ("/etc", "ro","/etc"),
            ("/etc/hosts", "ro","/etc/hosts"),
            ("/opt", "ro","/opt"),
            # ("/ide-init-bin", "ro","/ide-init-bin"),
            #("/init-bin", "ro","/init-bin"),
            #("/osscmd", "ro","/osscmd"),
            #("/pai-extension", "ro","/pai-extension"),
            ("/root", "ro","/root"),
            ("/run", "ro","/run"),
            ("/var", "ro","/var"),
            #("/lib/libsysconf-alipay.so","ro","/lib/libsysconf-alipay.so"),
            ("/dev", "rw","/dev"),
            ("/proc", "rw","/proc"),
            ("/sys", "rw","/sys"),
            ("/tmp", "rw","/tmp"),
            # conda
            (self._config.conda_env, "ro",self._config.conda_env),
            
            
        ]
        self.write_permit = [
            ("/root/.cache/pip/wheels", "/root/.cache/pip/wheels"),  # 宿主机 -> sandbox
          
        ]
   
        mount_cmds=[]
        for src,perm,tg in self.mount_points:
            if not os.path.exists(src):
                continue
            ttg=os.path.join(self._config.root_dir,tg.lstrip('/'))
            if not os.path.exists(ttg):
                os.makedirs(ttg,exist_ok=True)
            
            if perm=="ro":
                mount_cmds.append(f"mount --bind -o ro {src} {ttg}")
            else:
                mount_cmds.append(f"mount --bind {src} {ttg}")

        for src,tg in self.write_permit:
            if not os.path.exists(src):
                continue
            ttg=os.path.join(self._config.root_dir,tg.lstrip('/'))
            mount_cmds.append(f"mount --bind {src} {ttg}")
        self._ps1='SHELLPS1PREFIX'
        ps1 = self._ps1.replace("'", "'\"'\"'")  # 处理单引号以安全嵌入单引号字符串

        chroot_cmd = (
            f"exec chroot {self._config.root_dir} /bin/bash --noprofile --norc -c "
            f"'export USER=${{USER:-$(whoami)}}; export PS1=\"{ps1}\"; exec /bin/bash'"
        )

        full_cmd = ' && '.join(mount_cmds + [chroot_cmd])
        mount_ns_cmd = f"unshare --mount sh -c \"{full_cmd}\""
        full_cmd=mount_ns_cmd
        #write this command to a shell script under root_dir/start_sandbox.sh
        # script_path = os.path.join(self._config.root_dir, "start_sandbox.sh")
        # with open(script_path, "w", encoding="utf-8") as f:
        #     f.write("#!/bin/bash\n")
        #     f.write(full_cmd + "\n")
        # os.chmod(script_path, 0o755)
        run_shell_cmd=f"{full_cmd}\n"
        return run_shell_cmd


    
    def _pre_install_commands(self):
        """Run pre-install commands for each bundle."""
        
        for bundle in self._config.bundles:
            
            # self.env["PATH"] = f"{self.abs_tool_path}/{bundle.path.name}/bin:" + self.env["PATH"]
            env_path_cmd=f"export PATH={self.abs_tool_path}/{bundle.path.name}/bin:$PATH"

          
            cmds = [env_path_cmd] 
            
            if (bundle.path / "install.sh").exists():
                cmds.append(f"cd {self.abs_tool_path}/{bundle.path.name} && source install.sh")
            cmds.append(f"chmod +x {self.abs_tool_path}/{bundle.path.name}/bin/*")
           
            asyncio.run(self.runtime.run_in_session(BashAction(command="\n".join(cmds), timeout=100, check='raise')))
    

    def _install_commands(self):
        """Run install commands for each bundle. (Only chmod and export PATH here)"""
        
        for bundle in self._config.bundles:
            
            # self.env["PATH"] = f"{self.abs_tool_path}/{bundle.path.name}/bin:" + self.env["PATH"]
            env_path_cmd=f"export PATH={self.abs_tool_path}/{bundle.path.name}/bin:$PATH"

          
            cmds = [env_path_cmd] 
            
            if (bundle.path / "install.sh").exists() and bundle.path.name =='registry':
                cmds.append(f"cd {self.abs_tool_path}/{bundle.path.name} && source install.sh")
               
            cmds.append(f"chmod +x {self.abs_tool_path}/{bundle.path.name}/bin/*")
           
            asyncio.run(self.runtime.run_in_session(BashAction(command="\n".join(cmds), timeout=100, check='raise')))
        
       
    def create_env_dir(self,tg_dir,cahced_python_path,python_version=3.10):
        """Create a virtual environment directory using a cached Python path.

        Attributes:
            tg_dir (str): Target directory for the virtual environment.
            cahced_python_path (str): Path to the cached Python installation.
            python_version (float, optional): Python version to use. Defaults to 3.10.
        """
        #create venv based on the cached python path
        # path look like: cahced_python_path/python_version/miniconda3/bin/python
        # use subprocess to create venv
        print('tg_dir',tg_dir)
        import subprocess
        venv_cmd=f"{cahced_python_path}/{python_version}/miniconda3/bin/python -m venv {tg_dir}"
        subprocess.run(venv_cmd,shell=True,check=True)
        #print(f"Created venv in {tg_dir} using cached python at {cahced_python_path}")

    def safe_copytree(self,src, dst):
        from pathlib import Path
        #delete dst except python executables under /bin/
        import os
        import shutil
        dst_path = Path(dst)
        if dst_path.exists():
            for item in dst_path.iterdir():
                #if is bin 
                if item.is_dir() and item.name == 'bin':
                    for bin_item in item.iterdir():
                        if bin_item.name=='python' or bin_item.name =='python3' or bin_item.name ==f'python{self.py_version}':
                            # skip python executables
                            print(f"Skipping busy file: {bin_item.name}")
                            continue
                        if bin_item.is_file():
                            bin_item.unlink()
                elif item.is_symlink():
                    pass
                elif item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                else:
                    print(f"Unknown file type: {item}")


        # # copy all folders from src to dst except bin
        for item in Path(src).iterdir():
            
            if item.is_dir() and item.name == 'bin':
                # copy bin folder contents except python executables
                dest_bin = dst_path / item.name
                dest_bin.mkdir(parents=True, exist_ok=True)
                for bin_item in item.iterdir():
                    if bin_item.name=='python' or bin_item.name =='python3' or bin_item.name ==f'python{self.py_version}':
                        # skip python executables
                        print(f"Skipping busy file: {bin_item.name}")
                        continue
                    dest_bin_item = dest_bin / bin_item.name
                    if bin_item.is_dir():
                        shutil.copytree(bin_item, dest_bin_item)
                    else:
                        shutil.copy2(bin_item, dest_bin_item)
            else:
                dest = dst_path / item.name
                if item.is_symlink():
                    pass
                elif item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

    def additional_cmd(self,cmd_list=[]):
        """Run additional commands in the sandbox environment.
        Attributes:
            cmd_list (list, optional): List of commands to run. Defaults to [].
            if cmd_list is empty, use our default commands. (We use a proxy and set pip cache dir)
        """
        if len(cmd_list) ==0 :
            cmd_list = ['''export http_proxy="http://192.168.1.94:8080"''',
            '''export https_proxy="http://192.168.1.94:8080"''',
            '''export HTTP_PROXY="http://192.168.1.94:8080"''',
            '''export HTTPS_PROXY="http://192.168.1.94:8080"''',
            '''unset PIP_CONSTRAINT''',
            '''export USER=${USER:-$(whoami)}''',
            "export PS1='SHELLPS1PREFIX'",
            "export PIP_CACHE_DIR=/path/to/new/cache",
            "mkdir -p /_pip_cache"
            # 'unset CONDA_BUILD_SYSROOT',
            # 'unset _CONDA_PYTHON_SYSCONFIGDATA_NAME',
            # 'export CONDA_BUILD=0',
            # '''export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v 'compiler_compat' | paste -sd ':' -)"'''
            ]
        res=asyncio.run(self.runtime.run_in_session(BashAction(command="\n".join(cmd_list), timeout=12, check='raise')))
    
    def build_env(self,env):
        """Builds the virtual environment for the sandbox deployment.
        Attributes:
            env: The environment object for tracking hooks.
        """
        self.additional_cmd(self._config.cmd_list)
        Venv_Time_Record={}
        venv_cp_record={}
        ven_install_record={}
        env_name=self.git_id + '/' + self.version + '/' + self._config.ds.get('image_name','default')
        # git_folder='.'.join(instance_id.split('.')[:2])
        source_env_dir=os.path.join(self.shared_venv,env_name,'venv')
        self.env_dir_venv=self.root_dir+'/'+source_env_dir.lstrip('/')
        self.abs_venv_dir=source_env_dir
        env._chook.on_getting_testspec()
        if self._config.data_type=="swebench":
            self.test_spec=get_test_specs_from_ds(ds=self.ds,env_path=self.abs_venv_dir,env=env)
        tar_path = source_env_dir + ".tar.gz"
        install_repo_data={}
        if self._config.force_rebuild:
            if os.path.exists(tar_path):
                #delete the tar file to force rebuild
                os.remove(tar_path)
                
        if os.path.exists(tar_path):
            
            #directly copy it to self.root_dir/source_env_dir
            env._chook.on_copying_shared_venv()
            print(f"Copying shared venv from {source_env_dir} to {self.env_dir_venv}")
            venv_start_time = time.time()
            tar_extract(tar_path, self.env_dir_venv, threads=2)
            venv_end_time = time.time()
            venv_cp_record={"venv_copy_duration": venv_end_time - venv_start_time,
                            "venv_copy_start_time":venv_start_time,
                            "venv_copy_end_time":venv_end_time}
            #We need to ensure the venv is usable, so we check the venv 
            assert os.path.exists(self.env_dir_venv), f"Failed to extract venv from {tar_path}"
            linkp=source_env_dir
            # 虚拟环境的 Python 路径
            if os.name == "nt":
                self.venv_python = os.path.join(self.env_dir_venv, "Scripts", "python.exe")
                self.venv_pip = os.path.join(self.env_dir_venv, "Scripts", "pip.exe")
                self.venv_bin = os.path.join(self.env_dir_venv, "Scripts")
            else:
                self.venv_python = os.path.join(linkp, "bin", "python")
                self.venv_pip = os.path.join(linkp,"bin", "pip")
                self.venv_bin = os.path.join(linkp, "bin")
            # self.env["PATH"] = self.venv_bin + os.pathsep + self.env["PATH"]
            
            # self.env["VIRTUAL_ENV"] = os.path.dirname(self.venv_bin)  
            # self.env["PYTHONPATH"] = os.pathsep.join(list(self.needed_packages)) + os.pathsep + self.env.get("PYTHONPATH", "")
            env._chook.on_running_Pathcmds()
            Path_cmd=f"export PATH={self.venv_bin}:$PATH && export VIRTUAL_ENV={os.path.dirname(self.venv_bin)}"
            asyncio.run(self.runtime.run_in_session(BashAction(command=Path_cmd, timeout=30, check='raise')))
            install_repo_start_time = time.time()
            if not os.path.exists(self.cached_git_dir):
                # pip install -e . to install the repo to current venv
                res = asyncio.run(self.runtime.run_in_session(BashAction(command=f"cd /{self.git_folder} && source {self.venv_bin}/activate && pip install -e .", timeout=1000, check='raise')))
            install_repo_end_time = time.time()
            install_repo_data={"install_repo_duration": install_repo_end_time - install_repo_start_time,
                            "install_repo_start_time":install_repo_start_time,
                            "install_repo_end_time":install_repo_end_time}
        else:
            # create a new venv in source_env_dir
            print(f"Creating new venv in { source_env_dir}")
            
            self.create_env_dir(source_env_dir,self._config.conda_env,python_version=self.py_version)
            copytree_via_tar(source_env_dir, self.env_dir_venv, dirs_exist_ok=True,cached=False)
            linkp=source_env_dir
            print("New venv created")
            # 虚拟环境的 Python 路径
            if os.name == "nt":
                self.venv_python = os.path.join(self.env_dir_venv, "Scripts", "python.exe")
                self.venv_pip = os.path.join(self.env_dir_venv, "Scripts", "pip.exe")
                self.venv_bin = os.path.join(self.env_dir_venv, "Scripts")
            else:
                self.venv_python = os.path.join(linkp, "bin", "python")
                self.venv_pip = os.path.join(linkp,"bin", "pip")
                self.venv_bin = os.path.join(linkp, "bin")
            # self.env["PATH"] = self.venv_bin + os.pathsep + self.env["PATH"]
            
            # self.env["VIRTUAL_ENV"] = os.path.dirname(self.venv_bin)  
            # self.env["PYTHONPATH"] = os.pathsep.join(list(self.needed_packages)) + os.pathsep + self.env.get("PYTHONPATH", "")
            Path_cmd=f"export PATH={self.venv_bin}:$PATH && export VIRTUAL_ENV={os.path.dirname(self.venv_bin)}"
            asyncio.run(self.runtime.run_in_session(BashAction(command=Path_cmd, timeout=120, check='raise')))
            print('path_cmd',Path_cmd)
            #install needed packages
            res=self.install_env()
            
            self._pre_install_commands()
         
            
            #after install, cover it to source_env_dir
            if not self.delete_after_create:
                print(f"covering new venv to shared location {source_env_dir}")
                os.makedirs(os.path.dirname(source_env_dir), exist_ok=True)
                # We want to tar pack the source_env_dir to make the copy faster next time
                
                
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(self.env_dir_venv, arcname="")
                
            
            #delete source_env_dir
            shutil.rmtree(source_env_dir,ignore_errors=True)
        
        #check if self.cached_git_dir exist
        if (not os.path.exists(self.cached_git_dir) and not self.delete_after_create and self._config.cache_git) or self._config.force_rebuild:
            # pack testbed to self.cached_git_dir via tar
            testbed_dir=os.path.join(self.root_dir,self.git_folder.lstrip('/'))
            # copytree_via_tar(testbed_dir, self.cached_git_dir, dirs_exist_ok=True)
            print(f"Caching git repo to {self.cached_git_dir}")
            os.makedirs(os.path.dirname(self.cached_git_dir), exist_ok=True)
            with tarfile.open(self.cached_git_dir, "w:gz") as tar:
                tar.add(testbed_dir, arcname="")
        Venv_Time_Record['venv_cp_record']=venv_cp_record 
        Venv_Time_Record['install_repo_data']=install_repo_data
        return Venv_Time_Record
    def install_env(self):
        """Installs the eval environment for the sandbox deployment."""
        if self._config.data_type=="swebench":
            
            install_script_content=self.test_spec.setup_env_script+ ' \n ' +self.test_spec.install_repo_script
            
            

            # install_script_content="cd"
        elif self._config.data_type=="swesmith":
            install_cmd= get_install_commands_wrapper(self.ds)
            install_script_content = "\n".join(
                [
                   "#!/bin/bash",
                    # "set -uxo pipefail",
                    f"cd /{self.git_folder}",
                    'which pip',
                    f"source {self.venv_bin}/activate",
                    "pip install pytest",
                ] + install_cmd
            ) + "\n"
        else:
            install_script_content='cd ..'
       
        res=asyncio.run(self.runtime.run_in_session(BashAction(command=install_script_content, timeout=700, check='raise')))
        
        self.reset()
        return res.output

    def add_hook(self, hook: DeploymentHook):
        self._hooks.add_hook(hook)

    @classmethod
    def from_config(cls, config: SandboxDeploymentConfig) -> Self:
        return cls(**config.__dict__)

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive. The return value can be
        tested with bool().

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None:
            return IsAliveResponse(is_alive=False, message="Runtime is None.")
        return await self._runtime.is_alive(timeout=timeout)
    def create_sandbox(self):
        """Creates necessary directories for the sandbox."""
        os.makedirs(self._config.root_base, exist_ok=True)
        os.makedirs(self._config.shared_venv, exist_ok=True)
    async def start(self):
        """Starts the runtime."""
        # we want to add loop try to avoid the failure of starting the runtime

        max_loop=10
        loop_count=0
        while self._runtime is None and loop_count < max_loop:
            try:
                loop_count+=1
                self._runtime = LocalRuntime(logger=self.logger)
            
            except Exception as e:
                self.logger.error(f"Failed to start runtime: {e}")
                await asyncio.sleep(1)
        if self._runtime is None:
            raise DeploymentNotStartedError("Failed to start runtime after multiple attempts.")
        

      
    def post_init(self,env):
        """Post-initialization steps for the sandbox deployment.
        Attributes:
            env: The environment object for tracking hooks.
        """
        # self.mount()
        venv_build_start_time = time.time()
        venv_build_record = self.build_env(env)
        venv_build_end_time = time.time()
        venv_build_data={"venv_build_duration": venv_build_end_time - venv_build_start_time,
                        "venv_build_start_time":venv_build_start_time,
                        "venv_build_end_time":venv_build_end_time,
                        "venv_build_record":venv_build_record}

        env._chook.on_install_env_started()
        tool_install_start_time = time.time()
        self._install_commands()
        tool_install_end_time = time.time()
        tool_install_data={"tool_install_duration": tool_install_end_time - tool_install_start_time,
                            "tool_install_start_time":tool_install_start_time,
                            "tool_install_end_time":tool_install_end_time}
        env._chook.on_reset()
       
        self.reset()
        return {"venv_build_data":venv_build_data,"tool_install_data":tool_install_data}
    def reset(self):
        """Resets the sandbox environment to a clean state."""
        reset_cmd=['set +eux','set +o pipefail',"export USER=${USER:-$(whoami)} && export PS1='SHELLPS1PREFIX'"]
        

        
        res=asyncio.run(self.runtime.run_in_session(BashAction(command="\n".join(reset_cmd), timeout=12, check='raise')))
       
    async def stop(self):
        """Stops the runtime."""
        async def _close_runtime(runtime):
            with contextlib.suppress(Exception):
                await runtime.close()

        asyncio.create_task(_close_runtime(self._runtime))
        self.unmount()
        os.system(f"rm -rf {self.root_dir}")
    def unmount(self):
        pass
    @property
    def runtime(self) -> LocalRuntime:
        """Returns the runtime if running.

        Raises:
            DeploymentNotStartedError: If the deployment was not started.
        """
        if self._runtime is None:
            raise DeploymentNotStartedError()
        return self._runtime

    def pre_check(self):
        """The function to run the eval script before agent starts to ensure the test environment is ready (optional)."""
        if self._config.data_type=="swebench":
            self.setup_env_swebench()
        elif self._config.data_type=="swesmith":
            self.setup_env_swesmith()
            self.reset_swesmith_tests()

    def setup_env_swebench(self):
        """Sets up the evaluation environment for Swebench datasets."""
        eval_script_content=self.test_spec.eval_script
        # 目标路径
        script_path = Path(self.root_dir) / "run_tests.sh"

        # 确保目录存在
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # 写文件
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(eval_script_content)
        asyncio.run(self.runtime.run_in_session(BashAction(command="chmod +x /run_tests.sh && python3 -m pip install chardet", timeout=120, check='raise')))
    def setup_env_swesmith(self):
        """Sets up the evaluation environment for Swesmith datasets."""
        test_command, _ = get_test_commands_wrapper(self.ds)
        

        eval_script_content = "\n".join(
            [
                "#!/bin/bash",
                # "set -uxo pipefail",
                f"source {self.venv_bin}/activate",
                f"cd /{self.git_folder}",
                f": '>>>>> Start Test Output'",
                test_command,
                f": '>>>>> End Test Output'",
            ]
        ) + "\n"
        
        # 目标路径
        script_path = Path(self.root_dir) / "run_tests.sh"

        # 确保目录存在
        script_path.parent.mkdir(parents=True, exist_ok=True)

        # 写文件
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(eval_script_content)

        
        
        
        # self.run_command("chmod +x /run_tests.sh && python -m pip install chardet")
        asyncio.run(self.runtime.run_in_session(BashAction(command=f"chmod +x /run_tests.sh && python -m pip install chardet", timeout=120, check='raise')))

    
    
    def reset_swesmith_tests(self):
        """Resets the test files for Swesmith datasets to their original state."""
        f2p_files = list(set([x.split("::", 1)[0] for x in self.ds[FAIL_TO_PASS]]))
        p2p_files = list(set([x.split("::", 1)[0] for x in self.ds[PASS_TO_PASS]]))
        all_files = list(set(f2p_files + p2p_files))
        all_files = [f for f in all_files if 
             os.path.basename(f).startswith('test_') and os.path.basename(f).endswith('.py') or
             os.path.basename(f).endswith('_test.py')]
        commit_id ='origin/main'
        reset_command = (
            f'cd "/{self.git_folder}" && '
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout {commit_id} -- "{{}}" 2>/dev/null'
        )
        asyncio.run(self.runtime.run_in_session(BashAction(command=reset_command, timeout=60, check='raise')))

    def parse_logs(self, log_output: str) -> dict:
        """Parses the log output based on the repository type."""
        return parse_log_fn(f"{self.git_folder}")(log_output)
    def get_logs_eval(
        self, test_spec , content
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Attributes:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
       
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

      
        # Get status map of evaluation results
        # Get status map of evaluation results
        content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        # content = content.split(test_cmd)[-1]
        self.logger.info(f"using swebench log_parser for repo: {repo}")
        return log_parser(content, test_spec), True
    def _calculate_reward_swebench(self, get_test_output=False, timeout: int = 500):
        """Calculates the reward for Swebench datasets."""
        
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        
        if self.ds['instance_id'] in instance_to_skip:
            return 1.0,{},{},''
        self.setup_env_swebench()
        if self.ds['repo']=='pytest-dev/pytest':
            from configparser import ConfigParser
            p = self.root_dir+'/'+self.git_folder+'/'+'tox.ini'
            if os.path.exists(p):
                cp = ConfigParser()
                cp.read(p)
                if cp.has_section('pytest'):
                    cp.set('pytest', 'minversion', '0.0')
                
                with open(p, 'w') as f:
                    cp.write(f)

         
            p = self.root_dir+'/'+self.git_folder+'/'+'pyproject.toml'
            if os.path.exists(p):
                import tomli
                import tomli_w

                # 读取 pyproject.toml
                with open(p, "rb") as f:
                    data = tomli.load(f)

                # 修改 pytest 的 minversion
                if "tool" in data and "pytest" in data["tool"] and "ini_options" in data["tool"]["pytest"]:
                    data["tool"]["pytest"]["ini_options"]["minversion"] = "0.0"
                
                # 写回文件
                with open(p, "wb") as f:
                    tomli_w.dump(data, f)

        out= asyncio.run(self.runtime.run_in_session(BashAction(command="/run_tests.sh", timeout=timeout, check='raise'))).output
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        #print('eval_status_map',eval_status_map)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        
        
        eval_type = EvalType.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS \
        else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=eval_type
        )
        
        
        from swebench.harness.grading import compute_fail_to_pass,compute_pass_to_pass
        #
        skip_key_list=instance_map(instance_id=self.ds['instance_id'],repo=self.ds['repo'])
        
        def check_key_list(key, key_list):
            
            for k in key_list:
                
                if k in key: 
                    return True
            return False
        
        # if key in eval_ref[FAIL_TO_PASS], del it
        report[PASS_TO_PASS]["failure"] = [key for key in report[PASS_TO_PASS]["failure"] if not check_key_list(key, skip_key_list)]
        report[FAIL_TO_PASS]["failure"] = [key for key in report[FAIL_TO_PASS]["failure"] if not check_key_list(key, skip_key_list)]
    
        f2p = compute_fail_to_pass(report)
        p2p = compute_pass_to_pass(report)
        if self.ds['repo']=='pylint-dev/pylint':
            report[PASS_TO_PASS]["failure"]=report[PASS_TO_PASS]["failure"][16:]
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, f2p,p2p,out
        return success,f2p,p2p,out
    def _calculate_reward_swesmith(self, get_test_output=False, timeout: int = 300):
        """Calculates the reward for Swesmith datasets."""
        self.reset_swesmith_tests()
        self.setup_env_swesmith()
        output= asyncio.run(self.runtime.run_in_session(BashAction(command="/run_tests.sh", timeout=timeout, check='raise'))).output
        
        
        #output2= asyncio.run(self.runtime.run_in_session(BashAction(command="cat /run_tests.sh", timeout=timeout, check='raise'))).output
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
    def _calculate_reward(self,p2p=1,f2p=1):
        """Calculates the reward based on the dataset type."""
        # p2p and f2p: how many tests we allow to fail in pass_to_pass and fail_to_pass
        # if in this range, we still give reward 1.0
        timeout= self._config.eval_timeout
        def allowed_failures_reward(td, allowed):
            if allowed<0:
                return 1.0
            #deep copy the test_dict
            test_dict = copy.deepcopy(td)
            unique_tests = set(test_dict.keys())
            for test_name, status in test_dict.items():
                if (status == 'PASSED' or status=="SKIPPED") and test_name in unique_tests:
                    unique_tests.remove(test_name)
            
            failures = len(unique_tests)
            if failures <= allowed:
                return 1.0
            else:
                return 0.0
        if self._config.data_type=='swesmith':
            try:
                original_reward,fail2pass_dic,pass2pass_dic,output=self._calculate_reward_swesmith(timeout=timeout)
            except Exception as e:
                self.logger.error(f"Error in calculating reward swesmith: {e}")
                return 0.0,{},{},''
            #delte run_tests.sh
            script_path = Path(self.root_dir) / "run_tests.sh"
            if script_path.exists():
                script_path.unlink()
            return original_reward,fail2pass_dic,pass2pass_dic,output
            return allowed_failures_reward(fail2pass_dic,f2p)*allowed_failures_reward(pass2pass_dic,p2p),fail2pass_dic,pass2pass_dic,output
        elif self._config.data_type=='swebench':
            try:
                success,f2p,p2p,output=self._calculate_reward_swebench(timeout=timeout)
            except Exception as e:
                self.logger.error(f"Error in calculating reward swebench: {e}")
                return 0.0,{},{},''
            #delete run_tests.sh
            script_path = Path(self.root_dir) / "run_tests.sh"
            # if script_path.exists():
            #     script_path.unlink()
                
            return int(success),{'f2p':f2p},{'p2p':p2p},output
        else:
            raise ValueError(f"Unknown data type: {self._config.data_type}")
    
    def read_file(self,path, encoding=None, errors=None):
  
        path=os.path.join(self.root_dir, path.lstrip("/"))  
        p = Path(path)
        return p.read_text(encoding=encoding, errors=errors)
    def write_file(self, path: str, content: str, encoding: str = "utf-8", errors: str | None = None) -> None:
        
        # 拼接为容器内的绝对路径  
        path = os.path.join(self.root_dir, path.lstrip("/"))
        p = Path(path)

        # 确保目录存在
        p.parent.mkdir(parents=True, exist_ok=True)

        # 写入内容
        p.write_text(content, encoding=encoding, errors=errors)

    def extract_freetype_tarball(self,tarball_path, build_dir):
        """Extracts a tarball into the specified build directory within the sandbox."""
      
        import tarfile
        from pathlib import Path
        tarball_path = Path(tarball_path).resolve()
        build_dir=os.path.join(self.root_dir, build_dir.lstrip("/"))
        build_dir = Path(build_dir).resolve()
        

        if not tarball_path.exists():
            raise FileNotFoundError(f"找不到压缩包: {tarball_path}")
        if not build_dir.exists():
            #create build dir
            build_dir.mkdir(exist_ok=True)
        #print(f"正在解压 {tarball_path} 到 {build_dir} ...")

        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=build_dir)

       

    def upload(self, local_source, sandbox_target):
        
        if not os.path.exists(local_source):
            raise FileNotFoundError(f"not exist: {local_source}")

        # 拼接沙盒中的真实目标路径
        real_target_path = os.path.join(self.root_dir, sandbox_target.lstrip('/'),os.path.basename(local_source))

        # 确保目标目录存在（对于文件，确保它的上级目录存在）
        if os.path.isdir(local_source):
            os.makedirs(os.path.dirname(real_target_path), exist_ok=True)
            # 如果目标已存在同名目录，清理或合并，这里先清理
            if os.path.exists(real_target_path):
                shutil.rmtree(real_target_path)
            copytree_via_tar(local_source, real_target_path)
        else:
            # 是文件
            os.makedirs(os.path.dirname(real_target_path), exist_ok=True)
            shutil.copy2(local_source, real_target_path)

        return real_target_path
    def download(self, sandbox_source, target_dir):
        """download a file or directory from the sandbox"""
        
        real_source_path = os.path.join(self.root_dir, sandbox_source.lstrip('/'))

        if not os.path.exists(real_source_path):
            raise FileNotFoundError(f"not exist: {real_source_path}")

    
        os.makedirs(target_dir, exist_ok=True)

        if os.path.isfile(real_source_path):
            # 是文件，复制到目标目录
            target_path = os.path.join(target_dir, os.path.basename(real_source_path))
            shutil.copy2(real_source_path, target_path)
        elif os.path.isdir(real_source_path):
            # 是目录，递归复制整个目录
            target_path = os.path.join(target_dir, os.path.basename(real_source_path))
            if os.path.exists(target_path):
                # 如果已有同名目录，可选择清理或合并，这里先清理
                shutil.rmtree(target_path)
            copytree_via_tar(real_source_path, target_path)
        else:
            raise ValueError(f"不支持的源路径类型: {real_source_path}")

        return target_path
    
    def get_patch(self,dir='/res.patch') -> str:
        """
        Get the diff of the current state of the repository.
        """
        max_num_tries = 2
        num_tries=0
        for i in range(max_num_tries):
            try:
                output=asyncio.run(self.runtime.run_in_session(BashAction(command=f"cd /{self.git_folder} && git add -A && git diff --cached > {dir}", timeout=60, check='raise'))).output
                break
            except Exception as e:
                num_tries+=1
                self.logger.error(f"Failed to get patch, try {num_tries}/{max_num_tries}: {e}")
                if num_tries==max_num_tries:
                    raise e
                asyncio.run(asyncio.sleep(1))
        # output, _ = self.run("git diff")
        return output
    

    def close(self):
        """delete sandbox"""
        try:
            if self._runtime is not None:
                asyncio.run(self.stop())
            
        except Exception as e:
            pass

       

    def __del__(self):
        # 自动清理
        try:
            self.close()
        except Exception as e:

            raise e
