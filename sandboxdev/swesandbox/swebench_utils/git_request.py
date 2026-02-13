import os
import subprocess
import requests as real_requests
from requests.models import Response
from swebench.harness.constants import SWE_BENCH_URL_RAW
# 本地缓存目录
LOCAL_REPO_BASE = "/tmp/swe_repos"

def _run_git_cmd(cmd, cwd=None):
    """运行 git 命令并输出错误信息"""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\n{e}")

def _get_raw_file_via_git(url: str) -> Response:
    """
    从 raw.githubusercontent.com URL 通过本地 Git 获取文件内容（浅克隆版本）。
    URL 格式: https://raw.githubusercontent.com/<owner>/<repo>/<commit>/<filepath>
    """
    parts = url.replace(SWE_BENCH_URL_RAW, "").split("/", 3)
    if len(parts) < 4:
        raise ValueError(f"Unexpected raw URL format: {url}")
    owner, repo, commit, filepath = parts
    repo_full = f"{owner}/{repo}"
    repo_url = f"https://github.com/{repo_full}.git"

    # 本地缓存路径: 避免冲突，用 owner__repo 命名
    local_repo_path = os.path.join(LOCAL_REPO_BASE, f"{owner}__{repo}__{commit}")
    os.makedirs(LOCAL_REPO_BASE, exist_ok=True)

    if not os.path.exists(local_repo_path):
        # 第一次：浅克隆到临时目录
        _run_git_cmd(["git", "init"], cwd=LOCAL_REPO_BASE)
        os.makedirs(local_repo_path, exist_ok=True)
        # 直接初始化并指定远程
        _run_git_cmd(["git", "-C", local_repo_path, "init"])
        _run_git_cmd(["git", "-C", local_repo_path, "remote", "add", "origin", repo_url])
        # 浅抓这个 commit 对应的数据
        try:
            _run_git_cmd(["git", "-C", local_repo_path, "fetch", "--depth", "1", "origin", commit])
        except RuntimeError:
            # 如果浅抓失败，说明 commit 不在默认引用，尝试全量 fetch
            _run_git_cmd(["git", "-C", local_repo_path, "fetch", "origin", commit])
        # 切换到对应 commit
        _run_git_cmd(["git", "-C", local_repo_path, "checkout", commit])
    else:
        # # 目录存在时，只抓指定 commit
        # try:
        #     _run_git_cmd(["git", "-C", local_repo_path, "fetch", "--depth", "1", "origin", commit])
        # except RuntimeError:
        #     _run_git_cmd(["git", "-C", local_repo_path, "fetch", "origin", commit])
        pass

    

    local_file_path = os.path.join(local_repo_path, filepath)
    r = Response()
    if not os.path.exists(local_file_path):
        r.status_code = 404
        r._content = b"File not found"
        r.encoding = "utf-8"
        return r

    with open(local_file_path, "rb") as f:
        r._content = f.read()

    r.status_code = 200
    r.encoding = "utf-8"
    return r

# 覆盖 requests.get
def get(url, *args, **kwargs):
    if url.startswith(SWE_BENCH_URL_RAW):
        return _get_raw_file_via_git(url)
    else:
        return real_requests.get(url, *args, **kwargs)

# 其它方法透传真正的 requests
post = real_requests.post
put = real_requests.put
delete = real_requests.delete
head = real_requests.head
options = real_requests.options
Session = real_requests.Session
Request = real_requests.Request
