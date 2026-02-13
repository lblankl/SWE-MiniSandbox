# registry.py
import json
import os
from pathlib import Path

class SimpleRegistry(dict):
    """
    一个简单的跨平台 registry 实现，用字典存键值对。
    可选持久化到 JSON 文件。
    """
    _storage_file = Path(os.environ.get("REGISTRY_STORAGE", "/tmp/registry_storage.json"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果有持久化文件，则加载
        if self._storage_file.exists():
            try:
                with open(self._storage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    super().update(data)
            except Exception as e:
                print(f"[registry] Warning: failed to load registry storage: {e}")

    def save(self):
        """持久化当前数据到文件"""
        try:
            with open(self._storage_file, "w", encoding="utf-8") as f:
                json.dump(self, f)
        except Exception as e:
            print(f"[registry] Warning: failed to save registry storage: {e}")

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # 自动保存
        self.save()

    def get(self, key, default=None):
        return super().get(key, default)


# 创建一个全局的 REGISTRY 对象，供外部直接 import 使用
registry = SimpleRegistry()

# 可以设置一些默认值（防止获取时报错）
registry.setdefault("MAX_WINDOW_EXPANSION_VIEW", "1000")
registry.setdefault("MAX_WINDOW_EXPANSION_EDIT_CONFIRM", "1000")
registry.setdefault("USE_FILEMAP", "false")
registry.setdefault("USE_LINTER", "false")
registry.setdefault("file_history", "{}")
registry.setdefault("LINT_COMMAND", "flake8 --isolated --select=F821,F822,F831,E111,E112,E113,E999,E902 {file_path}")
