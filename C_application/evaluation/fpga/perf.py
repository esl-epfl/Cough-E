import re
import matplotlib.pyplot as plt
import pandas as pd

imu_data = []
audio_data = []

with open("fpga.log", "r") as f:
    for line in f:
        if "IMU CLK CYCLE" in line:
            m = re.search(r"IMU CLK CYCLE (\d+) AT WIND (\d+)", line)
            if m:
                imu_data.append(
                    {"window": int(m.group(2)), "clk_cycle": int(m.group(1))}
                )
        elif "AUDIO CLK CYCLE" in line:
            m = re.search(r"AUDIO CLK CYCLE (\d+) AT WIND (\d+)", line)
            if m:
                audio_data.append(
                    {"window": int(m.group(2)), "clk_cycle": int(m.group(1))}
                )

df_imu = pd.DataFrame(imu_data)
df_audio = pd.DataFrame(audio_data)

imu_mean = df_imu["clk_cycle"].mean()
imu_std = df_imu["clk_cycle"].std()
audio_mean = df_audio["clk_cycle"].mean()
audio_std = df_audio["clk_cycle"].std()

print(f"IMU Mean: {imu_mean:.2f}, Std: {imu_std:.2f}")
print(f"AUDIO Mean: {audio_mean:.2f}, Std: {audio_std:.2f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(
    df_imu["window"], df_imu["clk_cycle"], marker="o", color="blue", label="IMU"
)
ax1.set_title("IMU CLK CYCLE per Window")
ax1.set_xlabel("Window")
ax1.set_ylabel("Clock Cycles")
ax1.grid(True)
ax1.legend()

ax2.plot(
    df_audio["window"],
    df_audio["clk_cycle"],
    marker="s",
    color="orange",
    label="AUDIO",
)
ax2.set_title("AUDIO CLK CYCLE per Window")
ax2.set_xlabel("Window")
ax2.set_ylabel("Clock Cycles")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("clk_cycles.png")
plt.close()
