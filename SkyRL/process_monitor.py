#!/usr/bin/env python3
import csv
import time
import psutil
import getpass
from datetime import datetime


def get_total_threads():
    """统计系统总线程数（所有进程的 threads）。"""
    total_threads = 0
    for p in psutil.process_iter(attrs=['num_threads'], ad_value=None):
        try:
            total_threads += p.info.get('num_threads') or 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total_threads


def get_user_process_thread_counts(username: str):
    """统计指定用户的进程数和线程总数。"""
    proc_count = 0
    thread_count = 0
    for p in psutil.process_iter(attrs=['username', 'num_threads'], ad_value=None):
        try:
            if p.info.get('username') == username:
                proc_count += 1
                thread_count += p.info.get('num_threads') or 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return proc_count, thread_count


def collect_metrics(interval: float = 1.0, output_csv: str = "metrics_log.csv"):
    username = getpass.getuser()
    fieldnames = [
        "timestamp",
        "total_processes",
        "total_threads",
        "user_processes",
        "user_threads",
        "mem_total_MB",
        "mem_used_MB",
        "mem_percent",
        "vms_total_MB",
        "vms_used_MB",
    ]

    # 如果文件不存在，则写 header；存在则追加
    try:
        f = open(output_csv, "x", newline="")
        new_file = True
    except FileExistsError:
        f = open(output_csv, "a", newline="")
        new_file = False

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()

    print(f"Start collecting metrics every {interval} seconds...")
    print(f"Writing to: {output_csv}")
    print(f"Current user: {username}")

    try:
        while True:
            now = datetime.now().isoformat(timespec="seconds")

            # 总进程数
            total_processes = len(psutil.pids())
            # 总线程数
            total_threads = get_total_threads()

            # 当前用户进程/线程
            user_procs, user_threads = get_user_process_thread_counts(username)

            # 内存
            vm = psutil.virtual_memory()
            mem_total_MB = vm.total / (1024 * 1024)
            mem_used_MB = vm.used / (1024 * 1024)
            mem_percent = vm.percent

            # 虚拟内存（swap）
            sm = psutil.swap_memory()
            vms_total_MB = sm.total / (1024 * 1024)
            vms_used_MB = sm.used / (1024 * 1024)

            row = {
                "timestamp": now,
                "total_processes": total_processes,
                "total_threads": total_threads,
                "user_processes": user_procs,
                "user_threads": user_threads,
                "mem_total_MB": mem_total_MB,
                "mem_used_MB": mem_used_MB,
                "mem_percent": mem_percent,
                "vms_total_MB": vms_total_MB,
                "vms_used_MB": vms_used_MB,
            }

            writer.writerow(row)
            f.flush()

            print(
                f"[{now}] procs={total_processes} thr={total_threads} | "
                f"user_procs={user_procs} user_thr={user_threads} | "
                f"mem={mem_used_MB:.0f}/{mem_total_MB:.0f} MB ({mem_percent:.1f}%) | "
                f"swap={vms_used_MB:.0f}/{vms_total_MB:.0f} MB"
            )

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        f.close()


if __name__ == "__main__":
    # 可根据需要调整 interval 和 output_csv
    collect_metrics(interval=1.0, output_csv="metrics_log.csv")
