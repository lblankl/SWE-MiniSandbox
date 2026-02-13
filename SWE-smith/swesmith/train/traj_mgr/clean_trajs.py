"""
Remove unnecessary files from the trajectories directory.

Usage: python swesmith/
"""

import argparse
import os


def main(traj_dir):
    assert traj_dir.startswith("trajectories"), (
        "This script can only be run on SWE-agent trajectories."
    )
    for folder in sorted(
        [x for x in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, x))]
    ):
        folder = os.path.join(traj_dir, folder)
        removed = 0
        for root, _, files in os.walk(folder):
            for file in files:
                if any(
                    [
                        file.endswith(ext)
                        for ext in [
                            ".config.yaml",
                            ".debug.log",
                            ".info.log",
                            ".trace.log",
                        ]
                    ]
                ):
                    if file == "run_batch.config.yaml":
                        continue
                    # Delete this file
                    os.remove(os.path.join(root, file))
                    removed += 1
        print(f"{folder}: Removed {removed} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "traj_dir",
        type=str,
        help="Path to the directory containing the trajectories.",
    )
    args = parser.parse_args()
    main(**vars(args))
