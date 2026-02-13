

from datasets import load_from_disk,load_dataset
import argparse
import datasets
import tqdm

# datap="/us3/yuandl/dataset/SWE-bench/SWE-smith/data"
# use argparse to get datap default : /us3/yuandl/dataset/SWE-bench/SWE-smith/data
parser = argparse.ArgumentParser(description="Process dataset path.")
parser.add_argument("--datap", type=str,  help="Path to the dataset.", default="/us3/yuandl/dataset/SWE-bench/SWE-smith/data")
# save_path default: /us3/yuandl/dataset/SWE-bench/SWE-smith-unique_images
parser.add_argument("--save_path", type=str, required=False, help="Path to save the unique images dataset.", default="/us3/yuandl/dataset/SWE-bench/SWE-smith/unique_images")
args = parser.parse_args()
datap=args.datap
dataset = load_dataset(datap,split='train')

print(dataset[0]['image_name'])
# We want to find unique images based on image_name. One instance for one image_name is enough.
unique_image_names = set()
unique_images = []
for item in tqdm.tqdm(dataset, desc="Finding unique images"):
    image_name = item['image_name']
    if image_name not in unique_image_names:
        unique_image_names.add(image_name)
        unique_images.append(item)
print(f"Total unique images: {len(unique_images)}")
# Save unique images to a new dataset
unique_dataset = datasets.Dataset.from_list(unique_images)
unique_dataset.save_to_disk(args.save_path)
print(f"Unique images dataset saved to {args.save_path}")