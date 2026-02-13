import argparse
import os
import yaml
""" ymal example content
instances_by_exit_status:
    passed:
    - sympy__sympy-22456
    - sympy__sympy-12481
    - sympy__sympy-15976
    - psf__requests-1724
    - sympy__sympy-19783
    - sympy__sympy-19954
    - sympy__sympy-13551
    failed:
    - sympy__sympy-22085
    ......
"""

# yaml list args eg  1.yaml 2.yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description="A script that processes YAML files."
    )
    parser.add_argument(
        "--yaml_files",
        type=str,
        nargs="+",
        help="List of YAML files to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_output.yaml",
        help="Output YAML file name.",
    )
    return parser.parse_args()

def main():

    # merge all yaml files into one
    args = parse_args()
    merged_content = {"instances_by_exit_status":{}}
    for yaml_file in args.yaml_files:
        with open(yaml_file, "r") as f:
            content = yaml.safe_load(f)
            for key, value in content["instances_by_exit_status"].items():
                if key not in merged_content["instances_by_exit_status"]:
                    merged_content["instances_by_exit_status"][key] = []
                merged_content["instances_by_exit_status"][key].extend(value)
    
    with open(args.output, "w") as f:
        yaml.dump(merged_content, f)
if __name__ == "__main__":
    main()
            