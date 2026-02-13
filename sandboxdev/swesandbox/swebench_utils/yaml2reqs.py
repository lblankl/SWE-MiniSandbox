#!/usr/bin/env python3
import yaml

def extract_pip_requirements(env_yml_str):
    # 读取 environment.yml
    data = yaml.safe_load(env_yml_str)

    requirements = []

    # 获取 dependencies 列表
    deps = data.get("dependencies", [])

    for dep in deps:
        if isinstance(dep, str):
            # 过滤掉明显不是 pip 包的依赖，比如 python=3.9、numpy 等可以直接保留
            if not dep.startswith("python"):
                requirements.append(dep)
        elif isinstance(dep, dict) and "pip" in dep:
            # 如果有 pip 部分，直接加入
            requirements.extend(dep["pip"])

    # 去重
    requirements = list(dict.fromkeys(requirements))

   

    return "\n".join(requirements)

sttr='''name: myenv
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy>=1.26,<1.27
  - pip
  - pip:
      - requests
      - pandas==2.2.0
      - flask
'''
# extract_pip_requirements(sttr)
