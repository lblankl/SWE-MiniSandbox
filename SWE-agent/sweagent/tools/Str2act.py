
import datasets
from pathlib import Path
from sweagent.tools.bundle import Bundle
from sweagent.tools.tools import ToolConfig,ToolHandler
from sweagent.tools.parsing import XMLFunctionCallingParser
pp="/ossfs/workspace/nas/ais_k8s_task_repo_507090/dataset/SWE-smith-trajectories/data"
dataset=datasets.load_dataset(pp,split='train')


d=dataset[0]['messages']
# bundles:
      # - path: tools/registry
      # - path: tools/edit_anthropic
      # - path: tools/review_on_submit_m
bundle=[
  "tools/registry",
  "tools/edit_anthropic",
  "tools/review_on_submit_m",
]
basepath=Path("/ossfs/workspace/nas/agent/SWE-agent")
bundles=[]
for p in bundle:
  b=Bundle(path=basepath / p)
  bundles.append(b)

parser=XMLFunctionCallingParser()
toolcfg=ToolConfig(bundles=bundles,parse_function=parser)
commands=toolcfg.commands
toolhandler=ToolHandler(toolcfg)
n=0
for trace in d:
  role=trace['role']
  content=trace['content']
  if role=='assistant':
    
    
    #print(content)
    think,action=parser({"message":content},commands)
    print('command')
    print(action)
    print("After")
    transformed=toolhandler.guard_multiline_input(action)
    print(transformed)
    n+=1
    if n>=4:
      break
