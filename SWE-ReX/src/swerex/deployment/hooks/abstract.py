class DeploymentHook:
    def on_custom_step(self, message: str): ...


class CombinedDeploymentHook(DeploymentHook):
    def __init__(self, hooks: list[DeploymentHook] | None = None):
        self._hooks = []
        for hook in hooks or []:
            self.add_hook(hook)

    def add_hook(self, hook: DeploymentHook):
        self._hooks.append(hook)

    def on_custom_step(self, message: str):
        for hook in self._hooks:
            hook.on_custom_step(message)
