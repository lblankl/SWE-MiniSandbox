#Smith data share env based on image_name
# our sandbox also share venv based on image_name so we first need to filter unique data items in smith datasets

import datasets
smith_path= "/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/smith-passed"
ds=datasets.load_from_disk(smith_path)
unique_image_names = set()
import tqdm


# only keey the first occurrence of each unique image_name   for item in ds:
unique_items = []
seen_image_names = set()
for item in tqdm.tqdm(ds, desc="Filtering unique image names"):
    if item['image_name'] not in seen_image_names:
        unique_items.append(item)
        seen_image_names.add(item['image_name'])
unique_ds = datasets.Dataset.from_list(unique_items)
output_path="/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/smith-unique-image-name"
unique_ds.save_to_disk(output_path)

#
import yaml
# /home/smith/out/run_batch_exit_statuses.yaml
yaml_file_path = '/home/smith/out/run_batch_exit_statuses.yaml'

# Load the YAML file 
"""
instances_by_exit_status:
    Uncaught BashIncorrectSyntaxError:
    - cool-RR__PySnooper.57472b46.combine_file__271yh19z
    Uncaught BlockingIOError:
    - agronholm__exceptiongroup.0b4f4937.combine_file__5f69j6zc
    failed:
"""
#We only want the passed
import datasets
smith_path= "/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/smith-passed"
ds=datasets.load_from_disk(smith_path)
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.safe_load(file)
passed_instances = yaml_content.get('instances_by_exit_status', {}).get('passed', [])   
passed_set = set(passed_instances)  
filtered_ds=ds.filter(lambda example: example['image_name'] in passed_set)
output_path="/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/smith-passed-filtered"
filtered_ds.save_to_disk(output_path)