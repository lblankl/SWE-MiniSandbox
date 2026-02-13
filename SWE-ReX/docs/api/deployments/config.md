# Deployment configuration objects

These configuration objects can be used to configure deployments.

For example:

```python
from swerex.deployment.config import DockerDeploymentConfig

config = DockerDeploymentConfig(image="python:3.11")
deployment = config.get_deployment()
```

:::swerex.deployment.config
    options:
        members_order: source
        show_root_heading: false
        show_root_toc_entry: false
        show_source: false
        parameter_headings: false