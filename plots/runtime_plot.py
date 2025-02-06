import os
import json
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Cairo")

# Directory containing the log files
log_dir = r"E:\Code\husky_ws\Scaffold\data\mt"

# Lists to store data for plotting
model_names = []
runtimes = []
num_elements = []

# Read all _log.json files in the directory
for filename in os.listdir(log_dir):
    if filename.endswith("_log.json") and not (filename.startswith("box4x4") or filename.startswith("one_tet") or filename.startswith("box2x2_noclamp")):
        file_path = os.path.join(log_dir, filename)
        with open(file_path, "r") as file:
            log_data = json.load(file)
            summary = log_data["summary"]
            model_name = filename.replace("_log.json", "")
            model_names.append(model_name)
            runtimes.append(summary["runtime"])
            
            # Get the number of elements from the corresponding _layer_0.json file
            layer_file_path = os.path.join(log_dir, model_name + "_layer_0.json")
            if os.path.exists(layer_file_path):
                with open(layer_file_path, "r") as layer_file:
                    layer_data = json.load(layer_file)
                    num_elements.append(len(layer_data["line"]))
            else:
                num_elements.append(0)  # Default to 0 if the file does not exist

# Combine model names, runtimes, and number of elements, and sort by runtime
sorted_data = sorted(zip(runtimes, model_names, num_elements))

# Unzip the sorted data
runtimes, model_names, num_elements = zip(*sorted_data)

# Set font to Arial
plt.rcParams["font.family"] = "Arial"

# Plotting
plt.figure(figsize=(10, 6))

plt.scatter(num_elements, runtimes, label='Runtime')

# Annotate each point with the model name
for i, model_name in enumerate(model_names):
    plt.annotate(model_name, (num_elements[i], runtimes[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Number of Elements')
plt.ylabel('Runtime (seconds)')
# plt.yscale('log')
# plt.title('Runtime for Each Model')
# plt.legend()
plt.xticks(num_elements, rotation=90)
plt.tight_layout()

# Save the plot as a PDF file
output_path = os.path.join(os.path.dirname(__file__), 'runtime_plot.pdf')
plt.savefig(output_path)

plt.show()