#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    font = {
        'family': 'DejaVu Sans',  # 'Helvetica',
        'weight': '300',
        'size': 18
    }

    plt.rc('font', **font)
    plt.rc("figure", figsize=[8, 4])

    no_scaling_file_name = "no_scaling.csv"
    bert_file_name = "scaling.csv"
    
    no_scaling = pd.read_csv(no_scaling_file_name)
    bert = pd.read_csv(bert_file_name)

    indices = np.arange(0, 3601)
    for i in range(9):
        indices[i*400:] = indices[i*400:] + 400

    x = np.arange(0, 3601)
    scaling = 1 / 1000

    price_per_second = 3.87 / 3600 # $ Azure NC6s v3

    throughput = bert["throughput"]
    throughput = throughput[indices]
    throughput = throughput[throughput > 10] # filter
    throughput = throughput.ewm(alpha=0.1).mean()

    # throughput_money = (throughput / price_per_second) * scaling
    num_workers = bert["num_workers"][indices]
    total_throughput = throughput * num_workers
    # throughput_money = throughput_money.ewm(alpha=0.1).mean()
    total_throughput = total_throughput.ewm(alpha=0.1).mean()

    no_scaling_throughput = no_scaling["throughput"]
    no_scaling_throughput = no_scaling_throughput[indices]
    no_scaling_throughput = no_scaling_throughput[no_scaling_throughput > 10] # filter
    no_scaling_throughput = no_scaling_throughput.ewm(alpha=0.1).mean()

    # no_scaling_throughput_money = (no_scaling_throughput / price_per_second) * scaling
    no_scaling_num_workers = no_scaling["num_workers"][indices]
    no_scaling_total_throughput = no_scaling_throughput * no_scaling_num_workers
    # no_scaling_throughput_money = no_scaling_throughput_money.ewm(alpha=0.1).mean()
    no_scaling_total_throughput = no_scaling_total_throughput.ewm(alpha=0.1).mean()

    fig, ax = plt.subplots(constrained_layout=True)
 
    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    red = "#d62728"
    ax.plot(x, total_throughput, c=blue, label="KungFu Throughput")
    ax.plot(x, no_scaling_total_throughput, ls="-.", c=orange, label="Baseline Throughput")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (examples/sec)")
    # ax.legend(loc="lower left")
    ax.set_xlim([0, 3600])
    # ax.set_ylim([7.5, 26])
    y_range = np.arange(0, 120, 20)
    ax.set_yticks(y_range)
    for y_start in y_range:
        ax.axhline(y=y_start,
                   xmin=0,
                   xmax=1,
                   alpha=0.7,
                   linewidth=0.7,
                   ls=':',
                   color='grey')

    twin_ax = ax.twinx()
    twin_ax.plot(x, num_workers, ls=":", c=green, label="KungFu #GPUs")
    twin_ax.plot(x, no_scaling_num_workers, ls="--", c=red, label="Baseline #GPUs")
    twin_ax.set_ylabel("#GPUs")
    # twin_ax.legend(loc="lower right")
    twin_ax.set_ylim([0, 10])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    # fig.tight_layout()
    plt.savefig("optimal_cluster_size.pdf")

    print("no scaling", str(no_scaling_throughput[-800:].mean()))
    print("KungFu", str(throughput[-800:].mean()))


if __name__ == "__main__":
    main()
