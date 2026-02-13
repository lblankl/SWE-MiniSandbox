
import argparse
import os
import json

import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="A simple argument parser example.")
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON data file"
    )
    parser.add_argument(
        "--yaml_dir",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=os.getcwd(),
        help="Directory to save output files"
    )
    args = parser.parse_args()

    json_path = args.json_path
    print(f"JSON Path provided: {json_path}")
    yaml_dir = args.yaml_dir
    yaml_dir = Path(yaml_dir)
    exit_statuses_path = yaml_dir
    submitted_folders = set()
    
    if exit_statuses_path.exists():
            import yaml

            exit_statuses = yaml.safe_load(exit_statuses_path.read_text())
            for status, inst_list in exit_statuses.get("instances_by_exit_status", {}).items():
                if status == "submitted":
                    for inst in inst_list:
                        
                        submitted_folders.add(inst)
                       
    else:
        raise Exception("no file")

    with open(json_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    instance_ids = {}


    for entry in data:
        if entry['instance_id'] in submitted_folders:
            instance_id = '_'.join(entry['instance_id'].split('_')[:-1])

            if instance_id not in instance_ids:
                instance_ids[instance_id] = [entry]
            else:
                instance_ids[instance_id].append(entry)
    uniq_instances = set(instance_ids.keys())
    
    print(f"Number of unique instances in JSON: {len(uniq_instances)}")
    print(f"Number of submitted instances: {len(submitted_folders)}")
   
    count_statistics = {"1":0,"2":0,"3":0,"4":0}
    for instance_id, entries in instance_ids.items():
        
        num_entries = len(entries)
        if str(num_entries) in count_statistics:
            count_statistics[str(num_entries)] += 1
        else:
            count_statistics[str(num_entries)] = 1
    print("Count statistics of entries per unique instance:")
    for count, num_instances in count_statistics.items():
        print(f"Instances with {count} entries: {num_instances}")

    # balance the dataset, no more than 2 entries per unique instance
    balanced_data = []
    for instance_id, entries in instance_ids.items():
        if len(entries) <= 2:
            balanced_data.extend(entries)
        elif len(entries) > 2 and len(entries) <= 4:
            balanced_data.extend(entries[:2])
    #shuffle the balanced data and choose 5000 entries
    import random
    random.shuffle(balanced_data)
    balanced_data = balanced_data[:5100]
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving {len(balanced_data)} entries to {output_path}")

    with open(output_path, 'w') as outfile:
        json.dump(balanced_data, outfile)


if __name__ == "__main__":
    main()

