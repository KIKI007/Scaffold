import os
import json
import matplotlib.pyplot as plt

# Path to the JSON file
json_file_path = r"E:\Code\husky_ws\Scaffold\data\mt\box3x3_log.json"

# Read the JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Extract the tr_size and runtime values and corresponding iteration numbers
tr_sizes = [entry["tr_size"] for entry in data["original"]]
runtimes = [entry["runtime"] for entry in data["original"]]
iterations = list(range(1, len(tr_sizes) + 1))

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Trust Region Size', color=color)
ax1.plot(iterations, tr_sizes, marker='o', linestyle='-', color=color, label='Trust Region Size')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Runtime (seconds)', color=color)  # we already handled the x-label with ax1
ax2.plot(iterations, runtimes, marker='o', linestyle='-', color=color, label='Runtime')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

# Save the plot as a PDF file
output_path = os.path.join(os.path.dirname(__file__), 'tr_size_runtime_evolution.pdf')
plt.savefig(output_path)

plt.title('Evolution of Trust Region Size and Runtime')
plt.show()