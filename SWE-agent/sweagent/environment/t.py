import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
# D:\Research\Agent\dev\SWE-MiniSandbox\SWE-agent\sweagent\environment\repo.py
# D:\Research\Agent\dev\SWE-MiniSandbox\zip
zip_dir = os.path.abspath(os.path.join(current_dir, "../../../zip"))
print("zip_dir",zip_dir)
#list all files in zip_dir
for f in os.listdir(zip_dir):
    print("zip file:",f)