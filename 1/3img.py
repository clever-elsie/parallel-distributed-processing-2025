#!/usr/bin/env python3
import subprocess
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- 設定 ---
SOURCE_FILE = "3.cpp"
EXE_FILE = "./3.out"
OUTPUT_TXT = "3.stdout.txt"
OUTPUT_PNG = "3.stdout.png"
BASE_CMD = ["g++", "-std=c++20", "-lpthread", "-fopenmp", SOURCE_FILE, "-o", EXE_FILE]

# 実験設定
# コンパイルオプション
compilations = [
    ("No Optimization (-O0)", ["-O0"]),
    ("Optimization (-O2)", ["-O2"])
]

# 実行モード (3.cppの引数に対応)
exec_modes = [
    "reduction",
    "manual_reduction_with_atomic",
    "atomic_vector",
    "std::mutex",
    "pthread_mutex"
]

# スレッド数
threads_list = [1, 2, 4]

def compile_code(flags):
    """コードをコンパイルする"""
    cmd = BASE_CMD + flags
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation Failed:\n{result.stderr}")
        sys.exit(1)

def run_executable(mode, thread_count):
    """実行ファイルを実行し、標準出力を返す"""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(thread_count)
    
    # ./3.out <mode>
    cmd = [EXE_FILE, mode]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Execution Failed for {mode} with {thread_count} threads:\n{result.stderr}")
        return ""
    return result.stdout

def parse_duration(output_text):
    """出力から duration time を抽出する"""
    # 出力例: "duration time 0.003706 sec"
    match = re.search(r"duration time ([0-9\.]+) sec", output_text)
    if match:
        return float(match.group(1))
    return None

def main():
    full_log = []
    # データ構造: results[comp_label][mode][thread_count] = time
    results = {} 

    # 1. コンパイルと実行のループ
    for comp_label, flags in compilations:
        compile_code(flags)
        
        # ログ用ヘッダー
        header = f"\n# {comp_label} でコンパイル\n"
        print(header.strip())
        full_log.append(header)
        
        results[comp_label] = {}
        for mode in exec_modes:
            results[comp_label][mode] = {}
            
            for t in threads_list:
                # ログ用コマンド文字列
                cmd_log_str = f"OMP_NUM_THREADS={t} {EXE_FILE} {mode}"
                
                # 実行
                output = run_executable(mode, t)
                
                # ログ記録
                # ログ形式を見やすく整形して追加
                log_entry = f"{cmd_log_str}\n{output}"
                full_log.append(log_entry)
                print(output.strip()) # 進行状況表示
                
                # パース
                duration = parse_duration(output)
                if duration is not None:
                    results[comp_label][mode][t] = duration
                else:
                    results[comp_label][mode][t] = 0.0

    # 2. テキストファイルへ保存
    with open(OUTPUT_TXT, "w") as f:
        f.writelines(full_log)
    print(f"Logs saved to {OUTPUT_TXT}")

    # 3. グラフ描画
    plot_graph(results)

def plot_graph(results):
    """結果をグラフ化して保存する"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    width = 0.15
    
    comp_labels = list(results.keys())
    
    # X軸の位置設定
    x = np.arange(len(threads_list))
    
    for idx, comp_label in enumerate(comp_labels):
        ax = axes[idx]
        data_by_mode = results[comp_label]
        
        # モードごとに棒グラフを描画
        for i, mode in enumerate(exec_modes):
            times = []
            for t in threads_list:
                val = data_by_mode.get(mode, {}).get(t, 0.0)
                # 対数グラフ用に0を微小値に置換
                if val <= 1e-9: 
                    val = 1e-9 
                times.append(val)
            
            # 棒の位置をずらす
            offset = (i - len(exec_modes)/2) * width + width/2
            # ラベルは凡例用にきれいにする
            clean_label = mode.replace("manual_reduction_with_atomic", "manual_red_atomic")
            ax.bar(x + offset, times, width, label=clean_label, zorder=3)

        ax.set_title(comp_label, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t} threads" for t in threads_list], fontsize=12)
        ax.set_xlabel('Thread Count', fontsize=12)
        ax.grid(axis='y', which='both', linestyle='--', alpha=0.7, zorder=0)
        ax.set_yscale('log') # 対数軸
        
        if idx == 0:
            ax.set_ylabel('Time (sec) [Log Scale]', fontsize=12)

    # 凡例 (左のグラフから取得して共通化)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, bbox_inches='tight')
    print(f"Graph saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()