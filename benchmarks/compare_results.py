import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_results(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_stats(data, metric):
    keys = sorted([int(k) for k in data.keys()])
    means = []
    stds = []
    for k in keys:
        if metric == 'throughput':
            # Throughput = concurrency / mean_latency
            latencies = [entry.get('e2e', 0) for entry in data[str(k)]]
            if latencies and np.mean(latencies) > 0:
                # Approximate throughput assuming full saturation: C / avg(latency)
                means.append(k / np.mean(latencies))
                stds.append(0) # Std of throughput is harder to estimate this way
            else:
                means.append(0)
                stds.append(0)
        else:
            values = [entry.get(metric, 0) for entry in data[str(k)]]
            means.append(np.mean(values))
            stds.append(np.std(values))
    return keys, means, stds

def plot_comparison():
    plt.style.use('dark_background')
    baseline_path = 'finalproject/mem0/benchmarks/mem0-benchmark-baseline-20/benchmark_results.json'
    specdec_path = 'finalproject/mem0/benchmarks/mem0-benchmarks-batch-specdec/benchmark_results.json'

    if not os.path.exists(baseline_path) or not os.path.exists(specdec_path):
        print("Error: Result files not found.")
        return

    baseline_data = load_results(baseline_path)
    specdec_data = load_results(specdec_path)

    metrics = ['e2e', 'e2e_inc_bg', 'throughput']
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 7))
    fig.patch.set_facecolor('#1e1e1e')
    
    colors = ['#00d4ff', '#ff007f'] # Cyan and Pink for high contrast

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('#1e1e1e')
        b_keys, b_means, b_stds = get_stats(baseline_data, metric)
        s_keys, s_means, s_stds = get_stats(specdec_data, metric)

        if metric == 'throughput':
            ax.plot(b_keys, b_means, label='Baseline', marker='o', color=colors[0], linewidth=2, markersize=8)
            ax.plot(s_keys, s_means, label='SpecDec (Batch)', marker='s', color=colors[1], linewidth=2, markersize=8)
            ax.set_ylabel('Throughput (req/s)', fontsize=12, color='gray')
        else:
            ax.errorbar(b_keys, b_means, yerr=b_stds, label='Baseline', fmt='-o', color=colors[0], 
                        capsize=5, linewidth=2, markersize=8, alpha=0.8)
            ax.errorbar(s_keys, s_means, yerr=s_stds, label='SpecDec (Batch)', fmt='-s', color=colors[1], 
                        capsize=5, linewidth=2, markersize=8, alpha=0.8)
            ax.set_ylabel('Latency (seconds)', fontsize=12, color='gray')
        
        ax.set_title(f'{metric.upper()}', fontsize=16, fontweight='bold', pad=20, color='white')
        ax.set_xlabel('Concurrency (Batch Size)', fontsize=12, color='gray')
        ax.set_xscale('log', base=2)
        ax.set_xticks(b_keys)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.1, color='white')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

    plt.suptitle('Mem0 Benchmark Comparison: Baseline vs Speculative Decoding', fontsize=22, fontweight='bold', y=1.05, color='white')
    plt.tight_layout()
    output_path = 'finalproject/mem0/benchmarks/comparison_graph.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"Enhanced graph saved to {output_path}")

if __name__ == "__main__":
    plot_comparison()
