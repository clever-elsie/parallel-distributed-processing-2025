#!/usr/bin/env python3
import subprocess
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 設定 ---
SOURCE_FILE = "2.cpp"
EXE_FILE = "./2.out"
OUTPUT_TXT = "2.stdout.txt"
OUTPUT_PNG = "2.stdout.png"
# Makefileを参考にしたベースのコンパイルコマンド
BASE_CMD = ["g++", "-std=c++20", "-lpthread", "-fopenmp", SOURCE_FILE, "-o", EXE_FILE]

# 実験設定
# (ラベル, コンパイルオプションフラグ)
compilations = [
  ("No Optimization (-O0)", ["-O0"]),
  ("Optimization (-O2)", ["-O2"])
]

# (環境変数の辞書, ラベル)
# Noneの場合は現在の環境変数(デフォルトのスレッド数)を使用
run_configs = [
  (None, "Default Threads"),
  ({"OMP_NUM_THREADS": "2"}, "2 Threads"),
  ({"OMP_NUM_THREADS": "4"}, "4 Threads")
]

def compile_code(flags):
  """コードをコンパイルする"""
  cmd = BASE_CMD + flags
  print(f"Compiling: {' '.join(cmd)}")
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"Compilation Failed:\n{result.stderr}")
    sys.exit(1)

def run_executable(env_vars=None):
  """実行ファイルを実行し、標準出力を返す"""
  env = os.environ.copy()
  if env_vars:
    env.update(env_vars)
  
  result = subprocess.run([EXE_FILE], capture_output=True, text=True, env=env)
  if result.returncode != 0:
    print(f"Execution Failed:\n{result.stderr}")
    return ""
  return result.stdout

def parse_output(output_text):
  """出力をパースして辞書形式で返す"""
  # 形式: "48 threads seq: 0.003791 sec."
  pattern = re.compile(r"(\d+) threads (\w+): ([0-9\.]+) sec\.")
  data = {} # {thread_count: {method: time}}
  
  for line in output_text.splitlines():
    match = pattern.search(line)
    if match:
      threads = int(match.group(1))
      method = match.group(2)
      time_val = float(match.group(3))
      
      if threads not in data:
        data[threads] = {}
      data[threads][method] = time_val
  return data

def main():
  full_log = []
  parsed_results = {} # {comp_label: data_dict}

  # 1. コンパイルと実行のループ
  for comp_label, flags in compilations:
    compile_code(flags)
    
    # ログ用ヘッダー
    header = f"\n# {comp_label} でコンパイル\n"
    print(header.strip())
    full_log.append(header)
    
    # このコンパイル設定での結果格納用
    current_comp_data = {} 

    for env_vars, run_label in run_configs:
      cmd_str = " ".join([f"{k}={v}" for k,v in env_vars.items()]) + " " + EXE_FILE if env_vars else EXE_FILE
      log_line = f"Running: {cmd_str}"
      # print(log_line) # 進捗表示が必要ならコメントアウトを外す
      
      output = run_executable(env_vars)
      full_log.append(output)
      print(output.strip())

      # パースして統合
      # run_executableごとにパースすると上書きなどの制御が面倒なので
      # 取得したテキスト片ごとにパースしてマージする
      partial_data = parse_output(output)
      for t_count, methods_dict in partial_data.items():
        if t_count not in current_comp_data:
          current_comp_data[t_count] = {}
        current_comp_data[t_count].update(methods_dict)

    parsed_results[comp_label] = current_comp_data

  # 2. テキストファイルへ保存
  with open(OUTPUT_TXT, "w") as f:
    f.writelines(full_log)
  print(f"Logs saved to {OUTPUT_TXT}")

  # 3. グラフ描画 (以前のコードをベースに動的に対応)
  plot_graph(parsed_results)

def plot_graph(results):
  """結果をグラフ化して保存する"""
  methods = ['seq', 'cri', 'cri_mutex', 'red', 'atomic']
  
  # コンパイル設定が2つある前提のレイアウト
  fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
  width = 0.15
  
  comp_labels = list(results.keys())
  
  for idx, label in enumerate(comp_labels):
    ax = axes[idx]
    data = results[label]
    
    # スレッド数をソートして取得 (例: [2, 4, 48])
    threads_list = sorted(data.keys())
    x = np.arange(len(threads_list))
    
    for i, method in enumerate(methods):
      times = []
      for t in threads_list:
        val = data[t].get(method, 0.0)
        # 対数グラフ用に0を微小値に置換
        if val == 0.0:
          val = 1e-9 
        times.append(val)
      
      offset = (i - len(methods)/2) * width + width/2
      ax.bar(x + offset, times, width, label=method, zorder=3)

    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} threads" for t in threads_list], fontsize=12)
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.7, zorder=0)
    ax.set_yscale('log')
    
    if idx == 0:
      ax.set_ylabel('Time (sec) [Log Scale]', fontsize=12)

  # 凡例 (左のグラフから取得して共通化)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=12)
  
  plt.tight_layout()
  plt.savefig(OUTPUT_PNG, bbox_inches='tight')
  print(f"Graph saved to {OUTPUT_PNG}")
  # plt.show() # 自動化のため表示はコメントアウト

if __name__ == "__main__":
  main()