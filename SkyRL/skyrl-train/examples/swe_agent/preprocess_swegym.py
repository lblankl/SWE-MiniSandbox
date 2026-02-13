"""
Preprocess the SWEBench dataset to SkyRL format
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/swe_gym_subset")
    #data range  for example 0 128
    parser.add_argument("--data_range", type=int, nargs=2, default=[0, 128])
    parser.add_argument("--source_dir", default="~/data/SWEBench")
    parser.add_argument("--eval_data_dir", default="~/data/SWE-bench_Verified")
    parser.add_argument("--load_from_disk", action="store_true")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = args.source_dir
    eval_data_source = args.eval_data_dir
    if args.load_from_disk:
        train_dataset = datasets.load_from_disk(
            data_source
        ).shuffle(42).select(range(args.data_range[0], args.data_range[1]))
        
    else:
        train_dataset = datasets.load_dataset(
            data_source,
            split="train",
          
        ).shuffle(42).select(range(args.data_range[0], args.data_range[1]))
    val_dataset = datasets.load_dataset(
        eval_data_source,
        split="test",
    
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source if split == "train" else eval_data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": example["problem_statement"],
                    }
                ],
                "env_class": "null",
                "instance": example,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

###### sh
# python /home/zeta/SWE/SWE/SkyRL/skyrl-train/examples/swe_agent/preprocess_swegym.py \
#     --output_dir /us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/rl_formatted \
#     --source_dir /us3/yuandl/SWE_resource/CentOs/D/SWE/datasets/smith-image_passed \
#     --eval_data_dir /us3/yuandl/dataset/SWE-bench/SWE-bench_Verified/data \
#     --load_from_disk --data_range 0 128