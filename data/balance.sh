source /home/zeta/SWE/miniconda3/bin/activate
base=/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/sandbox4000
output_yaml=$base/merged.yaml # merged exit status record
json_path=$base/res.jsonl # the reward=1 merged training jsonl data


python /home/zeta/SWE/SWE/data/balance_data.py --json_path $json_path \
--yaml_dir $output_yaml --output_dir /us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/sandbox4000

source /home/zeta/SWE/miniconda3/bin/activate
base=/us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/docker4000
output_yaml=$base/merged.yaml # merged exit status record
json_path=$base/res.jsonl # the reward=1 merged training jsonl data

# python /home/zeta/SWE/SWE/sh/data_gather/gather.py \
# --jsonl_files $base/0-1150/dataset.jsonl $base/1150-2000/dataset.jsonl $base/2000-4000/dataset.jsonl $base/4000-6000/dataset.jsonl $base/4000-6000-3/dataset.jsonl $base/6000-8000/dataset.jsonl $base/6000-8000-3/dataset.jsonl \
# --tg_filepath $json_path

# python /home/zeta/SWE/SWE/data/merge_yaml.py --output $output_yaml \
# --yaml_files $base/0-1150/run_batch_exit_statuses.yaml $base/1150-2000/run_batch_exit_statuses.yaml $base/2000-4000/run_batch_exit_statuses.yaml $base/4000-6000/run_batch_exit_statuses.yaml $base/4000-6000-3/run_batch_exit_statuses.yaml $base/6000-8000/run_batch_exit_statuses.yaml $base/6000-8000-3/run_batch_exit_statuses.yaml

python /home/zeta/SWE/SWE/data/balance_data.py --json_path $json_path \
--yaml_dir $output_yaml --output_dir $base