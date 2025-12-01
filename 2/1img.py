#!/usr/bin/env python3
import subprocess
import re
import matplotlib.pyplot as plt
import sys
import argparse
import time

# --- 設定 ---
MPI_SOURCE = "1.cpp"
MPI_EXE = "./1.out"
OUTPUT_PNG = "1.stdout.png"

EXEC_MODES = ["linear", "square", "ring", "tree", "hypercube"]
SIZES = [1, 2, 4, 8, 16, 24]
SLEEP_INTERVAL = 0.05
TIMEOUT_SEC = 5

def compile_program():
    print("Compiling MPI program...")
    cmd = ["mpic++", "-O2", "-std=c++20", MPI_SOURCE, "-o", MPI_EXE]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error compiling {MPI_SOURCE}:\n{res.stderr}")
        sys.exit(1)
    print("Compilation successful.")

def run_single_experiment(mode, size):
    cmd = ["mpirun", "-n", str(size), MPI_EXE, mode]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SEC)
        if res.returncode != 0: return None
        times = [int(m.group(1)) for m in re.finditer(r"Time:\s*(\d+)\s*us", res.stdout)]
        if not times: return None
        return sum(times) / len(times)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception:
        return None

def plot_graph(results, trials):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    for i, mode in enumerate(EXEC_MODES):
        sizes, times = [], []
        for size in SIZES:
            if size in results[mode]:
                sizes.append(size)
                times.append(results[mode][size])
        plt.plot(sizes, times, marker=markers[i], label=mode, linewidth=2)

    plt.title(f'MPI AllReduce Performance (Average of {trials} trials)', fontsize=16)
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Average Time (microseconds)', fontsize=12)
    plt.xticks(SIZES)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"\nGraph saved to {OUTPUT_PNG}")

def main():
    parser = argparse.ArgumentParser(description="Run MPI Benchmark with progress display.")
    parser.add_argument("trials", nargs="?", type=int, default=1, help="Number of trials")
    args = parser.parse_args()
    
    num_trials = args.trials
    start_total_time = time.time()
    
    compile_program()
    final_results = {mode: {} for mode in EXEC_MODES}
    
    print(f"Starting benchmark ({num_trials} trials per configuration)...")
    total_configs = len(EXEC_MODES) * len(SIZES)
    current_config = 0
    
    for mode in EXEC_MODES:
        for size in SIZES:
            current_config += 1
            base_msg = f"  [{current_config}/{total_configs}] Mode: {mode:<10} Size: {size:<2}"
            valid_runs, sum_time = 0, 0.0
            
            for i in range(num_trials):
                print(f"\r{base_msg} | Trial {i+1}/{num_trials} ... ", end="", flush=True)
                val = run_single_experiment(mode, size)
                if isinstance(val, float):
                    sum_time += val
                    valid_runs += 1
                elif val == "TIMEOUT":
                    time.sleep(1.0)
                time.sleep(SLEEP_INTERVAL)
            
            if valid_runs > 0:
                final_results[mode][size] = avg_time = sum_time / valid_runs
            print(f"\r{base_msg} | Avg: {avg_time:.2f} us (Valid: {valid_runs}/{num_trials})   ")

    plot_graph(final_results, num_trials)
    elapsed = time.time() - start_total_time
    print(f"Total execution time: {elapsed:.2f} sec")

if __name__ == "__main__":
    main()