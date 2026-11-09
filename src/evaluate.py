"""
Benchmarking and performance visualization module for SePar.
Generates comparative figures across agent scales vs PRIMAL2, SCRIMP, and LaCAM2.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_throughput_results(agents, throughputs_separ, throughputs_lacam, map_name="warehouse_small", save_dir="benchmarks"):
    """
    Reproduces Figure 5 from the SePar paper (Average Throughput vs. Agent Density).
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(agents, throughputs_separ, marker='^', color='green', linewidth=2, label='SePar (Ours)')
    plt.plot(agents, throughputs_lacam, marker='v', color='red', linestyle='--', linewidth=2, label='LaCAM2')

    plt.xscale('log', base=2)
    plt.xticks(agents, [f"$2^{int(np.log2(a))}$" for a in agents])
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel('Targets / Timestep', fontsize=12)
    plt.title(f'Average Throughput - {map_name}', fontsize=14)
    plt.grid(True, alpha=0.6)
    plt.legend(fontsize=11)

    out_path = os.path.join(save_dir, f'throughput_{map_name}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Benchmark throughput plot saved to: {out_path}")


def generate_paper_benchmarks(save_dir="benchmarks"):
    """
    Generates standardized comparative benchmark plots using empirical paper metric data.
    """
    os.makedirs(save_dir, exist_ok=True)
    agents = [32, 64, 128, 256, 512]

    # Figure 5a: Warehouse Small Throughput
    separ_5a = [0.32, 0.32, 0.31, 0.26, 0.11]
    lacam_5a = [0.31, 0.32, 0.31, 0.30, 0.21]
    primal2_5a = [0.29, 0.31, 0.11, 0.02, 0.02]
    scrimp_5a = [0.32, 0.32, 0.31, 0.02, 0.01]

    plt.figure(figsize=(9, 5.5))
    plt.plot(agents, separ_5a, marker='^', color='green', linewidth=2, label='SePar')
    plt.plot(agents, lacam_5a, marker='v', color='crimson', linestyle='--', linewidth=1.5, label='LaCAM2')
    plt.plot(agents, scrimp_5a, marker='s', color='darkorange', linestyle='-', linewidth=1.5, label='SCRIMP')
    plt.plot(agents, primal2_5a, marker='p', color='steelblue', linestyle='-', linewidth=1.5, label='PRIMAL2')

    plt.xscale('log', base=2)
    plt.xticks(agents, [f"$2^{int(np.log2(a))}$" for a in agents])
    plt.xlabel('Number of Agents', fontsize=12)
    plt.ylabel('Throughput (targets/step)', fontsize=12)
    plt.title('Benchmark: Average Throughput (Warehouse Small)', fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=11)

    out_file = os.path.join(save_dir, 'Full_Results_Fig5_Fig6.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Paper benchmark curves saved to: {out_file}")
