#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

# ここに実行結果のログを貼り付けてください（またはファイルから読み込むようにしてもOKです）
log_text = """
  [1/30] Mode: linear     Size: 1  | Avg: 24.41 us (Valid: 100/100)
  [2/30] Mode: linear     Size: 2  | Avg: 86.77 us (Valid: 100/100)
  [3/30] Mode: linear     Size: 4  | Avg: 93.85 us (Valid: 100/100)
  [4/30] Mode: linear     Size: 8  | Avg: 256.25 us (Valid: 100/100)
  [5/30] Mode: linear     Size: 16 | Avg: 488.89 us (Valid: 100/100)
  [6/30] Mode: linear     Size: 24 | Avg: 741.51 us (Valid: 100/100)
  [7/30] Mode: square     Size: 1  | Avg: 49.39 us (Valid: 100/100)
  [8/30] Mode: square     Size: 2  | Avg: 78.26 us (Valid: 100/100)
  [9/30] Mode: square     Size: 4  | Avg: 152.44 us (Valid: 100/100)
  [10/30] Mode: square     Size: 8  | Avg: 408.71 us (Valid: 100/100)
  [11/30] Mode: square     Size: 16 | Avg: 605.84 us (Valid: 100/100)
  [12/30] Mode: square     Size: 24 | Avg: 887.23 us (Valid: 100/100)
  [13/30] Mode: ring       Size: 1  | Avg: 28.73 us (Valid: 100/100)
  [14/30] Mode: ring       Size: 2  | Avg: 68.53 us (Valid: 100/100)
  [15/30] Mode: ring       Size: 4  | Avg: 86.81 us (Valid: 100/100)
  [16/30] Mode: ring       Size: 8  | Avg: 132.35 us (Valid: 100/100)
  [17/30] Mode: ring       Size: 16 | Avg: 301.50 us (Valid: 100/100)
  [18/30] Mode: ring       Size: 24 | Avg: 551.57 us (Valid: 100/100)
  [19/30] Mode: tree       Size: 1  | Avg: 6.03 us (Valid: 100/100)
  [20/30] Mode: tree       Size: 2  | Avg: 70.78 us (Valid: 100/100)
  [21/30] Mode: tree       Size: 4  | Avg: 57.49 us (Valid: 100/100)
  [22/30] Mode: tree       Size: 8  | Avg: 67.51 us (Valid: 100/100)
  [23/30] Mode: tree       Size: 16 | Avg: 149.11 us (Valid: 100/100)
  [24/30] Mode: tree       Size: 24 | Avg: 183.16 us (Valid: 100/100)
  [25/30] Mode: hypercube  Size: 1  | Avg: 7.75 us (Valid: 100/100)
  [26/30] Mode: hypercube  Size: 2  | Avg: 59.01 us (Valid: 100/100)
  [27/30] Mode: hypercube  Size: 4  | Avg: 55.53 us (Valid: 100/100)
  [28/30] Mode: hypercube  Size: 8  | Avg: 64.86 us (Valid: 100/100)
  [29/30] Mode: hypercube  Size: 16 | Avg: 109.58 us (Valid: 100/100)
  [30/30] Mode: hypercube  Size: 24 | Avg: 146.97 us (Valid: 100/100)
"""

OUTPUT_PNG = "1.stdout.png"
EXEC_MODES = ["linear", "square", "ring", "tree", "hypercube"]
SIZES = [1, 2, 4, 8, 16, 24]

def parse_log(text):
    data = {mode: {} for mode in EXEC_MODES}
    # 正規表現で Mode, Size, Avg を抽出
    pattern = re.compile(r"Mode:\s+(\w+)\s+Size:\s+(\d+)\s+\|\s+Avg:\s+([\d\.]+)\s+us")
    
    for line in text.strip().splitlines():
        match = pattern.search(line)
        if match:
            mode = match.group(1)
            size = int(match.group(2))
            avg_time = float(match.group(3))
            
            if mode in data:
                data[mode][size] = avg_time
    return data

def plot_graph(results):
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, mode in enumerate(EXEC_MODES):
        sizes = []
        times = []
        for size in SIZES:
            if size in results[mode]:
                sizes.append(size)
                times.append(results[mode][size])
        
        plt.plot(sizes, times, marker=markers[i % len(markers)], label=mode, linewidth=2)

    plt.title('MPI AllReduce Performance (From Logs)', fontsize=16)
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Average Time (microseconds)', fontsize=12)
    plt.xticks(SIZES)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG)
    print(f"Graph restored and saved to {OUTPUT_PNG}")

if __name__ == "__main__":
    data = parse_log(log_text)
    plot_graph(data)