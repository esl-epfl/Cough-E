import re
import matplotlib.pyplot as plt
import pandas as pd

records = []
current_window = None
current_model = None

window_re = re.compile(r"PROCESS WINDOW (\d+) WITH MODEL (\d+)")
kernel_re = re.compile(r"(IMU|AUDIO)\|([\w_]+) CLK CYCLE (\d+)")

with open("fpga_kernel.log", "r") as f:
    for line in f:
        win_match = window_re.search(line)
        if win_match:
            current_window = int(win_match.group(1))
            current_model = "IMU" if win_match.group(2) == "0" else "AUDIO"
            continue

        ker_match = kernel_re.search(line)
        if ker_match and current_window is not None:
            val = int(ker_match.group(3))
            if val < 500:
                continue
            records.append(
                {
                    "model": current_model,
                    "kernel": ker_match.group(2),
                    "window": current_window,
                    "cycles": val,
                }
            )

df = pd.DataFrame(records)

df_agg = df.groupby(["model", "kernel", "window"])["cycles"].sum().reset_index()

summary = (
    df_agg.groupby(["model", "kernel"])["cycles"]
    .agg(["mean"])
    .reset_index()
)
print(summary)
print(summary.to_latex(index=False, float_format="{:,.0f}".format, escape=True))

models = df_agg["model"].unique()
for model in models:
    df_model = df_agg[df_agg["model"] == model]
    kernels = df_model["kernel"].unique()

    plt.figure(figsize=(10, 6))
    for kernel in kernels:
        df_ker = df_model[df_model["kernel"] == kernel].sort_values("window")
        plt.plot(
            df_ker["window"], df_ker["cycles"], marker="o", label=f"{kernel}"
        )

    plt.title(f"{model} Per-Kernel Clock Cycles per Window")
    plt.xlabel("Window")
    plt.ylabel("Clock Cycles")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model.lower()}_kernel_perf.png")
    plt.close()
