"""Tests for per-sandbox /tmp isolation in chroot mode (issue #4).

``startup_old()`` used to bind-mount the host ``/tmp`` into every sandbox,
so concurrent sandboxes raced on shared paths (e.g. pytest basetemp
numbering). The fix mounts a per-sandbox tmpfs instead. Heavy optional
dependencies (swerex / r2egym / swebench / ...) are stubbed when missing,
so this runs with just pydantic + pytest.
"""

import importlib
import os
import re
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sandboxdev"))


# ── dependency stubbing ──────────────────────────────────────────────────────

class _AbstractDeployment:
    """Minimal stand-in usable as a base class."""

    def __init__(self, *args, **kwargs):
        pass


def _stub_module(name, **attrs):
    """Register a stub for ``name`` unless the real module is importable."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: MagicMock()  # PEP 562 fallback
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod


for _name, _attrs in [
    ("swerex", {}),
    ("swerex.deployment", {}),
    ("swerex.deployment.abstract", {"AbstractDeployment": _AbstractDeployment}),
    ("swerex.deployment.config", {}),
    ("swerex.deployment.hooks", {}),
    ("swerex.deployment.hooks.abstract", {}),
    ("swerex.exceptions", {"DeploymentNotStartedError": type("DeploymentNotStartedError", (Exception,), {})}),
    ("swerex.runtime", {}),
    ("swerex.runtime.abstract", {}),
    ("swerex.runtime.sandbox", {}),
    ("swerex.utils", {}),
    ("swerex.utils.log", {}),
    ("r2egym", {}),
    ("r2egym.repo_analysis", {}),
    ("r2egym.repo_analysis.execution_log_parser", {}),
    ("r2egym.swesmith", {}),
    ("r2egym.swesmith.utils", {}),
    ("r2egym.swesmith.constants", {}),
    ("swebench", {}),
    ("swebench.harness", {}),
    ("swebench.harness.constants", {}),
    ("swebench.harness.log_parsers", {}),
    ("swebench.harness.grading", {}),
    ("swebench.harness.dockerfiles", {}),
    ("swebench.harness.utils", {}),
    ("swebench.harness.test_spec", {}),
    ("swebench.harness.test_spec.javascript", {}),
    ("swebench.harness.test_spec.utils", {}),
    ("ray", {}),
    ("yaml", {}),
    ("requests", {}),
    ("requests.models", {}),
]:
    _stub_module(_name, **_attrs)

from swesandbox.sandbox_deployment import SandboxDeployment  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_deployment(root_dir, conda_env="/nonexistent-conda-env"):
    """Build a SandboxDeployment with just the state startup_old() needs."""
    deployment = object.__new__(SandboxDeployment)
    deployment._config = SimpleNamespace(root_dir=str(root_dir), conda_env=conda_env)
    return deployment


def _unshare_mount_argv():
    """Return an unshare argv prefix that can create mount namespaces here.

    Falls back to ``--map-root-user`` (user namespace) so the functional test
    also runs in unprivileged containers without CAP_SYS_ADMIN. Returns None
    if neither variant works.
    """
    for argv in (["unshare", "--mount"], ["unshare", "--map-root-user", "--mount"]):
        try:
            result = subprocess.run(
                argv + ["sh", "-c", "mount -t tmpfs tmpfs /mnt && echo ok"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0 and b"ok" in result.stdout:
                return argv
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


_UNSHARE_ARGV = _unshare_mount_argv()


# ── command-generation tests ─────────────────────────────────────────────────

def test_startup_old_does_not_bind_mount_host_tmp(tmp_path):
    deployment = _make_deployment(tmp_path / "sandbox")
    cmd = deployment.startup_old()
    assert not re.search(r"mount --bind (-o ro )?/tmp\s", cmd), (
        "host /tmp must not be bind-mounted into the sandbox: " + cmd
    )


def test_startup_old_mounts_private_tmpfs_on_tmp(tmp_path):
    root_dir = tmp_path / "sandbox"
    deployment = _make_deployment(root_dir)
    cmd = deployment.startup_old()
    assert f"mount -t tmpfs -o mode=1777 tmpfs {root_dir}/tmp" in cmd
    # the tmpfs must be mounted inside the private mount namespace
    assert cmd.startswith("unshare --mount")
    # the mount target must exist before the namespace is entered
    assert (root_dir / "tmp").is_dir()


def test_startup_old_tmpfs_mounted_before_chroot(tmp_path):
    root_dir = tmp_path / "sandbox"
    cmd = _make_deployment(root_dir).startup_old()
    assert cmd.index(f"mount -t tmpfs -o mode=1777 tmpfs {root_dir}/tmp") < cmd.index("chroot")


# ── functional isolation test ────────────────────────────────────────────────

@pytest.mark.skipif(_UNSHARE_ARGV is None, reason="requires unshare --mount + tmpfs privileges")
def test_concurrent_sandboxes_get_isolated_tmp(tmp_path):
    """Two concurrent namespaces using the generated tmpfs mount must not
    see each other's /tmp contents (the issue-#4 collision scenario)."""
    num_sandboxes = 4

    def run_sandbox(index):
        root_dir = tmp_path / f"sandbox{index}"
        cmd = _make_deployment(root_dir).startup_old()
        mount_cmd = re.search(r"mount -t tmpfs[^&]*", cmd).group(0).strip()
        # Inside the real sandbox, chroot makes {root_dir}/tmp appear as /tmp,
        # so writing a fixed filename here mimics colliding /tmp paths.
        marker = root_dir / "tmp" / "pytest-collision-marker"
        script = (
            f"{mount_cmd} && "
            f"echo sandbox_{index} > {marker} && "
            f"sleep 0.5 && "
            f"cat {marker}"
        )
        return subprocess.run(
            _UNSHARE_ARGV + ["sh", "-c", script],
            capture_output=True, text=True, timeout=30,
        )

    with ThreadPoolExecutor(max_workers=num_sandboxes) as executor:
        results = list(executor.map(run_sandbox, range(num_sandboxes)))

    for index, result in enumerate(results):
        assert result.returncode == 0, result.stderr
        # with a shared /tmp the overlapping writers would overwrite each other
        assert result.stdout.strip() == f"sandbox_{index}"

    # tmpfs lives only inside each namespace: nothing leaks to the host view
    for index in range(num_sandboxes):
        assert list((tmp_path / f"sandbox{index}" / "tmp").iterdir()) == []
