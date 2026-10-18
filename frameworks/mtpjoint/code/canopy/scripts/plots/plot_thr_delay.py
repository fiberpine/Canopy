"""
Read the log files and plot throughput vs delay.
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_family", choices=['synthetic' ,'real'])
    parser.add_argument("directories", nargs="+", help="one or more directory paths")
    args = parser.parse_args()

    assert len(args.directories) > 0, "You didn't provide any directories"

    traces_to_plot = {
        "synthetic": [
            "bump1000_12_24", "bump100_6_12", "bump500_24_12", "bump500_96_48", "mountain500_12_24", 
            "sawtooth1000_12_24", "sawtooth100_12_24", "sawtooth500_24_12", "sawtooth500_96_48", 
            "bump1000_24_48", "bump500_12_24", "bump500_48_96", "bump50_6_12", "mountain500_48_96", 
            "sawtooth1000_24_48", "sawtooth500_12_24", "sawtooth500_48_96", "sawtooth50_12_24"
        ],
        "real": ["ATT-LTE-driving-2016", "TMobile-LTE-short", "Verizon-LTE-short"]
    }

    # setup plot format
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(6, 4))

    colors = ["#82B366", "#D79B00", "#9673A6", "#6C8EBF", "#D6B656", "#B85450", "#BF5700"]
    markers = ["o", "P", "^", "s", "p", "d", "v"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (2, 1)), (0, (5, 1))]


    # Read the sum files.
    for index, directory in enumerate(args.directories):
        count = 0
        utilizations = []

        avg = 0
        p95 = 0
        util = 0
        total_drops = 0

        runs = list(map(lambda x: int(x.replace("run", "")), os.listdir(directory)))
        print(f"#{index}: Directory = {directory}. Found {len(runs)} runs.")
        runs = sorted(runs)
        for run in runs:
            # Read all files in directory/run{run} and choose the one which has trace in its name.
            run_avg = 0
            run_p95 = 0 
            run_util = 0 
            run_count = 0

            for file in list(filter(lambda x: os.path.isfile(os.path.join(f"{directory}/run{run}", x)),  os.listdir(f"{directory}/run{run}"))):
                to_plot = any([f"{t}.down" in file for t in traces_to_plot[args.trace_family]])
                
                # Check if any of the terms from traces_to_plot[args.trace_family] is in the file name.
                if not to_plot:
                    print(file, "XYZ")
                    continue

                if file.startswith("trimmedsum"):
                    trace_util, trace_avg, trace_p95, _ = read_sum_file(f"{directory}/run{run}/{file}")
                    if trace_util == 0:
                        print(f"Trace not long enough. Skipping {file}")
                        continue

                    avg += trace_avg
                    p95 += trace_p95
                    util += trace_util
                    count += 1
                    
                    run_avg += trace_avg
                    run_p95 += trace_p95
                    run_util += trace_util
                    run_count += 1
            
            run_avg /= run_count
            run_p95 /= run_count
            run_util /= run_count

            print(f"\tRun #{run}: avg={run_avg}, util={run_util}:")

        if directory.endswith("/"):
            label = directory[:-1]
        
        label = label.split("/")[-1]

        avg /= count
        p95 /= count
        util /= count
        print("Combined", avg, p95, util)
        
        plt.plot(
            avg,
            util,
            label=label,
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linestyle=styles[index % len(styles)],
            markersize=16,
        )

        # Draw a line from (avg, util) to (p95, util)
        plt.plot(
            [avg, p95],
            [util, util],
            color=colors[index%len(colors)],
            linestyle=styles[index%len(styles)],
            linewidth=4,
        )

    plt.xlabel("Average Queueing Delay (ms)", fontsize=24)
    plt.ylabel("Utilization (%)", fontsize=24)
    plt.grid()
    plt.legend(loc="lower right", ncol=2, columnspacing=0.5)
    plt.subplots_adjust(top=0.95, bottom=0.21, left=0.18, right=0.98)

    plt.savefig(f"thr_delay_{args.trace_family}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()