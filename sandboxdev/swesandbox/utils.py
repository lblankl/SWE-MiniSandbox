import subprocess
import threading
import os
import tarfile
import tempfile
import ray
import os
import math
import os
import socket
io_seperation_mb = 50.0  # 每 50MB 一个 io 单位

def get_file_size_mb(path: str) -> float:
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 ** 2)  # 转成 MiB

def get_tar_io_for_file(path: str) -> float:
    size_mb = get_file_size_mb(path)
    resource_name = "tar_io"
    this_ip = socket.gethostbyname(socket.gethostname())
    io_max=40
    for node in ray.nodes():
        # if node["Alive"] and node["NodeManagerAddress"] == this_ip:
        io_max = node["Resources"].get(resource_name,40)
    return min(max(1, math.ceil(size_mb / io_seperation_mb)),io_max)  # 最少 1 个 io 单位     


def tar_extract(tar_path, dst, threads: int = 4, ):
    """
    Unpack tar file to destination directory dst. We implement both local and remote (Ray) versions of I/O bounded logic.
    
    Attributes:
        tar_path: Path to the tar file.
        dst: Destination directory to extract files into.
        threads: Number of threads to use for extraction (not used in this implementation).
    """
    remote = ray.is_initialized()
    if remote:
        ref = tar_extract_remote(tar_path, dst)  # 立即返回一个 ObjectRef
        ray.get(ref) 
        
    else:
        tar_extract_local(tar_path, dst)
# def copytree_via_tar(src, dst, *, dirs_exist_ok=False, threads: int = 2):
#     """
#     用“打包 + 解压”的方式代替逐文件 shutil.copytree。
#     语义上尽量等价于:
#         shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)
#     优点：对于大量小文件的目录复制，通常比逐文件复制更快。
#     参数:
#         src: 源目录路径
#         dst: 目标目录路径
#         dirs_exist_ok: 与 shutil.copytree 一致
#             - False: 目标目录不能存在，否则抛 FileExistsError
#             - True: 允许目标已存在，在其中解压内容（可能覆盖/合并）
#     """
#     remote = ray.is_initialized()
#     if remote:
#         #copytree_via_tar_remote.remote(src, dst, dirs_exist_ok=dirs_exist_ok, threads=threads)
#         ref = copytree_via_tar_remote.remote(src, dst, dirs_exist_ok=dirs_exist_ok, threads=threads)
#         ray.get(ref)
#     else:
#         copytree_via_tar_local(src, dst, dirs_exist_ok=dirs_exist_ok, threads=threads)
def extract(tar_path, dst, threads: int = 4):
        """
        解压 tar 文件到目标目录 dst。
        """
    
        os.makedirs(dst, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dst)
# less than 10MB
@ray.remote(resources={"tar_io": float(10)/io_seperation_mb})
def extract_remote_10MB(tar_path, dst):
    extract(tar_path, dst)
# between 10MB and 50MB
@ray.remote(resources={"tar_io": float(50)/io_seperation_mb})
def extract_remote_50MB(tar_path, dst):
    extract(tar_path, dst)
# between 50MB and 100MB
@ray.remote(resources={"tar_io": float(100)/io_seperation_mb})
def extract_remote_100MB(tar_path, dst):
    extract(tar_path, dst)
# between 100MB and 200MB
@ray.remote(resources={"tar_io": float(200)/io_seperation_mb})
def extract_remote_200MB(tar_path, dst):
    extract(tar_path, dst)
# more than 200MB
@ray.remote(resources={"tar_io": float(500)/io_seperation_mb})
def extract_remote_large(tar_path, dst):
    extract(tar_path, dst)

def make_extract_remote(tar_io_need: float):
   @ray.remote(resources={"tar_io": tar_io_need})
   def extract(tar_path, dst, threads: int = 4):
        """
        解压 tar 文件到目标目录 dst。
        """
    
        os.makedirs(dst, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dst)
   return extract
# def make_extract_remote(tar_io_need: float):
#     if tar_io_need <=10:
#         return extract_remote_10MB
#     elif tar_io_need <=50:
#         return extract_remote_50MB
#     elif tar_io_need <=100:
#         return extract_remote_100MB
#     elif tar_io_need <=200:
#         return extract_remote_200MB
#     else:
#         return extract_remote_large

def tar_extract_remote(tar_path, dst):
    """
    解压 tar 文件到目标目录 dst。
    """
   
    tar_io_need = get_tar_io_for_file(tar_path)
    extract = make_extract_remote(tar_io_need)
    return extract.remote(tar_path, dst)

class SizeCapacityLimiter:
    """
    基于文件大小（MB）的并发容量控制：
      - total_capacity_mb: 允许“正在解压的文件大小之和”的最大值
    """
    def __init__(self, total_capacity_mb: float):
        self.total_capacity_mb = float(total_capacity_mb)
        self.used_capacity_mb = 0.0
        self.lock = threading.Lock()
        # 新增：当前活跃任务数
        self.active_tasks = 0
        self.cond = threading.Condition(self.lock)

    def acquire(self, need_mb: float):
        """阻塞直到有足够容量可以占用 need_mb"""
        need_mb = float(need_mb)
        with self.cond:
            # 如果需要的容量本身就超过总容量，可以直接抛异常或直接占满
            if need_mb > self.total_capacity_mb:
                # 这里选择直接占满（防止死锁）
                while self.used_capacity_mb > 0:
                    self.cond.wait()
                self.used_capacity_mb = need_mb
                self.active_tasks += 1
                return

            # 否则正常等待
            while self.used_capacity_mb + need_mb > self.total_capacity_mb:
                self.cond.wait()
            self.used_capacity_mb += need_mb
            self.active_tasks += 1
    def release(self, need_mb: float):
        """释放容量，并唤醒等待的线程"""
        need_mb = float(need_mb)
        with self.cond:
            self.used_capacity_mb -= need_mb
            if self.used_capacity_mb < 0:
                self.used_capacity_mb = 0
            self.active_tasks -= 1
            if self.active_tasks < 0:
                self.active_tasks = 0
            self.cond.notify_all()
    def get_active_tasks(self) -> int:
        """获取当前正在占用容量的任务数"""
        with self.cond:
            return self.active_tasks
    def get_used_capacity(self) -> float:
        """获取当前已使用的容量（MB）"""
        with self.cond:
            return self.used_capacity_mb

def get_file_size_mb(path: str) -> float:
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 ** 2)


# 全局一个容量限制器，比如最多同时解压 10,000 MB（约 10 GB）
TAR_CAPACITY_LIMITER = SizeCapacityLimiter(total_capacity_mb=200)

def tar_extract_local(tar_path: str, dst: str):
    """
    解压 tar 文件到目标目录 dst。
    通过 TAR_CAPACITY_LIMITER 控制：所有正在解压文件的大小总和 <= total_capacity_mb。
    """
    size_mb = get_file_size_mb(tar_path)

    # print("required size",size_mb,"MB")
    # print("total activate:", TAR_CAPACITY_LIMITER.get_active_tasks())
    # print("used cap:", TAR_CAPACITY_LIMITER.get_used_capacity())
    # 1. 申请容量（如果没空闲，会在这里阻塞等待）
    TAR_CAPACITY_LIMITER.acquire(size_mb)

    try:
        os.makedirs(dst, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dst)
    finally:
        # 2. 解压结束，释放容量
        TAR_CAPACITY_LIMITER.release(size_mb)


def copytree_via_tar(src, dst, *, dirs_exist_ok=False, cached=True):
    """
    用“打包 + 解压”的方式代替逐文件 shutil.copytree。
    语义上尽量等价于:
        shutil.copytree(src, dst, dirs_exist_ok=dirs_exist_ok)

    优点：对于大量小文件的目录复制，通常比逐文件复制更快。

    参数:
        src: 源目录路径
        dst: 目标目录路径
        dirs_exist_ok: 与 shutil.copytree 一致
            - False: 目标目录不能存在，否则抛 FileExistsError
            - True: 允许目标已存在，在其中解压内容（可能覆盖/合并）
    """
    
    src = os.fspath(src)
    dst = os.fspath(dst)

    if not os.path.isdir(src):
        raise NotADirectoryError(f"src is not a directory: {src}")

    # 行为对齐 shutil.copytree 的 dirs_exist_ok 逻辑
    if os.path.exists(dst):
        if not dirs_exist_ok:
            raise FileExistsError(f"Destination '{dst}' already exists")
        if not os.path.isdir(dst):
            raise NotADirectoryError(f"Destination '{dst}' exists and is not a directory")
    else:
        os.makedirs(dst, exist_ok=True)

    # 创建临时 tar 文件
    tar_name =src.replace("/","_")
    tar_path = f"/tmp/copytree_via_tar/{tar_name}.tar"
    if cached and os.path.exists(tar_path):
        if ray.is_initialized():
            ref = tar_extract_remote(tar_path, dst)
            ray.get(ref)
        else:
            tar_extract_local(tar_path, dst)
        return 
    else:
        # create tmp dir and file
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        open(tar_path, "w").close()

    try:
        # 打包 src 目录
        with tarfile.open(tar_path, "w") as tar:
            # 第二个参数 arcname="" 是为了让解压时把 src 的内容直接展开到 dst 下，
            # 而不是创建一层 src 目录。
            tar.add(src, arcname="")

        # 解压到 dst
        with tarfile.open(tar_path, "r") as tar:
            # extractall 不会自动创建 dst，要确保上面已创建
            tar.extractall(path=dst)

    finally:
        # 删除临时 tar
        if not cached:
            try:
                os.remove(tar_path)
            except FileNotFoundError:
                pass

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

import subprocess
import venv
import os


class EvaluationError(Exception):
    pass

def apply_patch(sandbox_root,git_folder,patch_str,instance_id,reverse=False):
    #write the patch to a patch file under sandbox_root
    patch_file = os.path.join(sandbox_root, "temp_patch.diff")
    with open(patch_file, "w") as f:
        f.write(patch_str)
    cwd=os.path.join(sandbox_root, git_folder)
    _apply_patch_local(instance_id, patch_file, cwd, reverse)
def _apply_patch_local(instance_id: str, patch_file: str, cwd: str, reverse: bool = False):
    """
    Apply a patch to local codebase in cwd using multiple fallback methods.
    :param instance_id: ID for logging/reporting
    :param patch_file: Patch file path (diff file)
    :param cwd: Directory of the repo to apply patch
    :param reverse: Reverse apply if True
    """
    apply_succeeded = False
    last_stdout = ""
    last_stderr = ""
    last_cmd = ""

    for base_cmd in GIT_APPLY_CMDS:
        # gold patch = bug patch, fix = revert
        if reverse:
            if base_cmd.startswith("patch"):
                cmd = base_cmd.replace("-i", "-R -i") + f" {patch_file}"
            else:  # git apply
                cmd = f"{base_cmd} --reverse {patch_file}"
        else:
            cmd = f"{base_cmd} {patch_file}"

        print(f"[INFO] Trying: {cmd}")
        last_cmd = cmd
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        last_stdout = proc.stdout
        last_stderr = proc.stderr

        if proc.returncode == 0:
            print(f"[SUCCESS] Patch applied with: {cmd} {cwd}")
            print(last_stdout)
            apply_succeeded = True
            #delete the patch file
            os.remove(patch_file)
            break
        else:
            print(f"[FAILED] Command: {cmd}")
            print(last_stdout)
            print(last_stderr)
            print("[INFO] Trying next method...")

    if not apply_succeeded:
        os.remove(patch_file)
        apply_failed_msg = f"Failed to apply patch for {instance_id} with all methods.\nLast command: {last_cmd}\nSTDERR:\n{last_stderr}"
        raise EvaluationError(apply_failed_msg)