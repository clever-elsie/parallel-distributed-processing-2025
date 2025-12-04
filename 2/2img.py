#!/usr/bin/env python3
import subprocess
import os
import re
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import sys

# --- 設定 ---
EXE_FILE = "../1/3.out"
OUTPUT_TXT = "2.stdout.txt"
OUTPUT_PNG = "2.stdout.png"
OUTPUT_EPS = "2.stdout.eps"
NUM_TRIALS = 5000  # 各設定の実行回数

# 実行コマンドの設定: (numactlコマンド, OMP_NUM_THREADS, ラベル)
CONFIGS = [
  (["numactl", "--physcpubind=0"], "1", "physcpubind=0, OMP_NUM_THREADS=1"),
  (["numactl", "--physcpubind=0"], "4", "physcpubind=0, OMP_NUM_THREADS=4"),
  (["numactl", "--cpunodebind=0"], "4", "cpunodebind=0, OMP_NUM_THREADS=4"),
  (["numactl", "--cpunodebind=0"], None, "cpunodebind=0, default threads"),
  (["numactl", "--cpunodebind=0", "--membind=1"], None, "cpunodebind=0, membind=1, default threads"),
  (["numactl", "--cpunodebind=1", "--membind=0"], None, "cpunodebind=1, membind=0, default threads"),
]

def run_once(numactl_cmd, omp_threads):
  """1回実行して実行時間を返す"""
  cmd = numactl_cmd + (["env", f"OMP_NUM_THREADS={omp_threads}", EXE_FILE] if omp_threads else [EXE_FILE])
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    return None
  match = re.search(r"duration time ([0-9\.]+) sec", result.stdout)
  return float(match.group(1)) if match else None

def run_config(numactl_cmd, omp_threads, label):
  """設定ごとに複数回実行して平均値を計算"""
  print(f"Running {NUM_TRIALS} trials: {label}")
  times = []
  for i in range(NUM_TRIALS):
    duration = run_once(numactl_cmd, omp_threads)
    if duration is not None:
      times.append(duration)
    if (i + 1) % max(NUM_TRIALS // 5, 1) == 0:
      print(f"  [{i+1}/{NUM_TRIALS}] Valid: {len(times)}/{i+1}")
  avg_time = sum(times) / len(times) if times else 0.0
  print(f"  Average: {avg_time:.6f} sec (Valid: {len(times)}/{NUM_TRIALS})\n")
  return avg_time, len(times)

def plot_graph(results):
  """結果をグラフ化"""
  labels = list(results.keys())
  times = [results[label] for label in labels]
  
  fig, ax = plt.subplots(figsize=(14, 8))
  x = np.arange(len(labels))
  bars = ax.bar(x, times, zorder=3)
  
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
  ax.set_ylabel('Time (sec) [Log Scale]', fontsize=12)
  ax.set_title(f'演習2-2の実行結果（numactl設定による比較、各{NUM_TRIALS}回の平均）', fontsize=14, fontweight='bold')
  ax.grid(axis='y', which='both', linestyle='--', alpha=0.7, zorder=0)
  ax.set_yscale('log')
  
  for bar, time in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
        f'{time:.6f}', ha='center', va='bottom', fontsize=9)
  
  plt.tight_layout()
  plt.savefig(OUTPUT_PNG, bbox_inches='tight', dpi=150)
  plt.savefig(OUTPUT_EPS, bbox_inches='tight')
  print(f"Graph saved to {OUTPUT_PNG} and {OUTPUT_EPS}")

def main():
  if not os.path.exists(EXE_FILE):
    print(f"Error: {EXE_FILE} not found. Please run 'make -C ../1 3.out' first.")
    sys.exit(1)
  
  results = {}
  log_lines = []
  
  for numactl_cmd, omp_threads, label in CONFIGS:
    avg_time, valid_count = run_config(numactl_cmd, omp_threads, label)
    results[label] = avg_time
    cmd_str = " ".join(numactl_cmd) + (f" env OMP_NUM_THREADS={omp_threads}" if omp_threads else "") + f" {EXE_FILE}"
    log_lines.append(f"{cmd_str}\nAverage: {avg_time:.6f} sec (Valid: {valid_count}/{NUM_TRIALS})\n\n")
  
  with open(OUTPUT_TXT, "w") as f:
    f.writelines(log_lines)
  print(f"Logs saved to {OUTPUT_TXT}")
  
  plot_graph(results)

if __name__ == "__main__":
  main()
