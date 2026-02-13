"""
Given a folder of SWE-agent trajectories, extracts the trajectories
and transforms them into a fine-tuning compatible jsonl format, namely...

[
  {
    "messages": [
      {
        "role": "system",
        "content": "system prompt (optional)"
      },
      {
        "role": "user",
        "content": "human instruction"
      },
      {
        "role": "assistant",
        "content": "model response"
      }
    ]
  },
  ...
]

"""
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from swebench.harness.constants import KEY_INSTANCE_ID, LOG_REPORT
from swesmith.constants import generate_hash
from swesmith.train.traj_mgr.utils import MAP_STYLE_TO_FUNC
from tqdm.auto import tqdm
from typing import Optional, Tuple


def process_single_trajectory(
    folder: str,
    traj_dir: Path,
    eval_dir: Path,
    transform_traj,
) -> Optional[Tuple[str, dict]]:
    """Process a single trajectory folder and return the result."""
    # if not (eval_dir / folder).exists():
    #     return None
    # if not (eval_dir / folder / LOG_REPORT).exists():
    #     return None

    try:
        
        pred_path = traj_dir / folder / f"{folder}.pred"
        #{"reward": 0, "test_out": "/ro ....
        pred_content = json.loads(pred_path.read_text())
        reward = pred_content.get("reward", 0)

        traj_path = traj_dir / folder / f"{folder}.traj"
        traj_orig = json.loads(traj_path.read_text())
        traj = transform_traj(traj_orig)
        traj[KEY_INSTANCE_ID] = folder
        traj["resolved"] = reward > 0
        if "replay_config" in traj_orig:
            traj["model"] = json.loads(traj_orig["replay_config"])["agent"]["model"][
                "name"
            ]
        traj["traj_id"] = f"{folder}.{generate_hash(str(traj_dir))}"
        traj["patch"] = traj_orig.get("patch", "")

        return (folder, traj)
    except Exception as e:
        print(f"Error processing folder {folder}: {e}")
        return None


def main(
    out_path: Path,
    traj_dir: Path,
    eval_dir: Path,
    style: str,
    workers: int,
):
    if style not in MAP_STYLE_TO_FUNC:
        raise ValueError(
            f"Style {style} not supported. Options: {list(MAP_STYLE_TO_FUNC.keys())}"
        )
    transform_traj = MAP_STYLE_TO_FUNC[style]

    folders = [x.name for x in traj_dir.iterdir() if x.is_dir()]
    print(f"Found {len(folders)} trajectory folders in {traj_dir}")

    

    # Process trajectories in parallel
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        # But we only want those that exists folder.traj 
        # future_to_folder = {
        #     executor.submit(
        #         process_single_trajectory, folder, traj_dir, eval_dir, transform_traj
        #     ): folder
        #     for folder in folders
        # }
        #  traj_dir /run_batch_exit_statuses.yaml we only want those submitted
        submitted_folders = set()
        exit_statuses_path = traj_dir / "run_batch_exit_statuses.yaml"
        if exit_statuses_path.exists():
            import yaml

            exit_statuses = yaml.safe_load(exit_statuses_path.read_text())
            for status, inst_list in exit_statuses.get("instances_by_exit_status", {}).items():
                if status == "submitted":
                    submitted_folders.update(inst_list)
        future_to_folder = {}
        for folder in folders:
            traj_path = traj_dir / folder / f"{folder}.traj"
            pred_path = traj_dir / folder / f"{folder}.pred"
            if traj_path.exists() and folder in submitted_folders and pred_path.exists():
                future = executor.submit(
                    process_single_trajectory,
                    folder,
                    traj_dir,
                    eval_dir,
                    transform_traj,
                )
                future_to_folder[future] = folder

        # Collect results as they complete
        # We only want resolved trajectories
        # for future in tqdm(
        #     as_completed(future_to_folder),
        #     total=len(folders),
        #     desc="Processing trajectories",
        # ):
        #     result = future.result()
        #     if result is not None:
        #         results.append(result)
        for future in tqdm(
            as_completed(future_to_folder),
            total=len(future_to_folder),
            desc="Processing trajectories",
        ):
            result = future.result()
            if result is not None and result[1].get("resolved", False):
                results.append(result)

    # Write results to file
    num_trajs = 0
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _, traj in results:
            f.write(json.dumps(traj) + "\n")
            num_trajs += 1

    print(f"Wrote {num_trajs} valid trajectories to {out_path.absolute()}")


if __name__ == "__main__":
    user = os.getenv("USER")

    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "-t",
        "--traj_dir",
        type=Path,
        required=False,
        help="Path to folder containing SWE-agent trajectories. Default: trajectories/{user}/",
        default=f"trajectories/{user}/",
    )
    arg_parser.add_argument(
        "-e",
        "--eval_dir",
        type=Path,
        required=False,
        default="logs/run_evaluation/",
        help="Path to folder containing evaluation results. Default: logs/run_evaluation/",
    )
    arg_parser.add_argument(
        "-s",
        "--style",
        type=str,
        required=False,
        default="xml",
        choices=list(MAP_STYLE_TO_FUNC.keys()),
        help="Style of the trajectories",
    )
    arg_parser.add_argument(
        "-o",
        "--out_path",
        type=Path,
        required=False,
        default="./dataset.jsonl",
        help="Path to output file",
    )
    arg_parser.add_argument(
        "-w",
        "--workers",
        type=int,
        required=False,
        default=min(32, os.cpu_count() + 4),
        help="Maximum number of worker threads. Default: min(32, os.cpu_count() + 4)",
    )
    args = arg_parser.parse_args()
    main(**vars(args))
