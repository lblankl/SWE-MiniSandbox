#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

def load_log(csv_path):
    """读取 CSV 日志，返回 (times, counts)"""
    times = []
    counts = []

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Log file not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["timestamp"]
            cnt_str = row["process_count"]
            try:
                dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                cnt = int(cnt_str)
            except Exception:
                continue

            times.append(dt)
            counts.append(cnt)

    return times, counts

def plot_processes(csv_path, output_png=None, show=True):
    times, counts = load_log(csv_path)

    if not times:
        print("No data in log file.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(times, counts, marker="o", linestyle="-", linewidth=1)

    plt.xlabel("Time")
    plt.ylabel("Process Count")
    plt.title(f"Process Count Over Time\n{csv_path}")

    plt.grid(True)
    plt.tight_layout()

    if output_png:
        output_png = Path(output_png)
        plt.savefig(output_png, dpi=150)
        print(f"Saved figure to {output_png}")

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # 示例：默认读取最近的一个日志文件
    log_dir = Path("monitor_logs")
    csv_files = sorted(log_dir.glob("proc_monitor_*.csv"))
    if not csv_files:
        print("No log files found in monitor_logs/")
    else:
        latest = csv_files[-1]
        print(f"Plotting latest log: {latest}")
        plot_processes(latest, output_png="proc_monitor_latest.png", show=True)
