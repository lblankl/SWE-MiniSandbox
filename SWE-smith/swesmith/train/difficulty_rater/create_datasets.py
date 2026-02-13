"""
Purpose: Create difficulty train / test datasets from SWE-bench Verified annotations of task difficulty.

Usage:
python train/difficulty_rater/create_datasets.py

NOTE: Please include the follwing files in the same directory when running this script:
- ensembled_annotations_public.csv
- samples_with_3_annotations_public.csv
"""

import json
import pandas as pd

from collections import Counter
from datasets import load_dataset
from swebench.harness.constants import KEY_INSTANCE_ID

PROMPT_SYSTEM = """Below I have given you information about a GitHub pull request. The information includes
the problem statement describing the bug and the patch representing the changes made that
successfully resolves the issue. Please categorize the difficulty of the original task based
on this information. There are 4 levels of difficulty you can choose from:

* <15 min fix
* 15 min - 1 hour
* 1-4 hours
* >4 hours"""

PROMPT_INSTANCE = """### Input:
**Problem Statement**
{problem_statement}

**Solution Patch**
{patch}

**Response**
"""

if __name__ == "__main__":
    sweb = load_dataset("SWE-bench/SWE-bench")
    sweb_map = {x[KEY_INSTANCE_ID]: x for x in sweb["test"]}
    ensembled = pd.read_csv("ensembled_annotations_public.csv")
    samplesw3 = pd.read_csv("samples_with_3_annotations_public.csv")

    df = ensembled[[KEY_INSTANCE_ID, "difficulty"]]
    test_df = df.sample(frac=0.2, random_state=42)
    train_df = df.drop(test_df.index)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    for pair in [
        ("difficulty_train.jsonl", train_df),
        ("difficulty_test.jsonl", test_df),
    ]:
        distribution = []
        with open(pair[0], "w") as f:
            for row in pair[1].itertuples(index=False, name=None):
                inst = sweb_map[row[0]]
                label = row[1]
                if label == ">4 hours":
                    label = "1-4 hours"
                messages = {
                    "messages": [
                        {"role": "system", "content": PROMPT_SYSTEM},
                        {"role": "user", "content": PROMPT_INSTANCE.format(**inst)},
                        {"role": "assistant", "content": label},
                    ]
                }
                distribution.append(label)
                f.write(json.dumps(messages) + "\n")

        print(f"{pair[0]} distribution:")
        for k, v in Counter(distribution).items():
            print(f"* {k}: {v} ({round(v * 100 / len(distribution), 2)}%)")

    with open("difficulty_train.jsonl") as f:
        check = [json.loads(x) for x in f.readlines()]
    print(len(check))
    with open("difficulty_test.jsonl") as f:
        check = [json.loads(x) for x in f.readlines()]
    print(len(check))
