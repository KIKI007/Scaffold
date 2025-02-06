import os
import json
import numpy as np

# Directory containing the log files
log_dir = r"E:\Code\husky_ws\Scaffold\data\mt"

# Lists to store data for the table
table_data = []

# Read all _log.json files in the directory
for filename in os.listdir(log_dir):
    if filename.endswith("_log.json") and not (filename.startswith("box2x2_noclamp") or filename.startswith("one_tet")):
        file_path = os.path.join(log_dir, filename)
        with open(file_path, "r") as file:
            log_data = json.load(file)
            summary = log_data["summary"]
            model_name = filename.replace("_log.json", "")
            
            # Get the number of elements from the corresponding _layer_0.json file
            layer_file_path = os.path.join(log_dir, model_name + "_layer_0.json")
            if os.path.exists(layer_file_path):
                with open(layer_file_path, "r") as layer_file:
                    layer_data = json.load(layer_file)
                    num_elements = len(layer_data["line"])
                    
                    # Calculate node valence average and standard deviation
                    node_valences = [len(adj) for adj in layer_data["adj"]]
                    valence_avg = np.mean(node_valences)
                    valence_std = np.std(node_valences)
            else:
                num_elements = 0
                valence_avg = 0
                valence_std = 0
            
            # Get the number of trust region iterations and total runtime
            trust_region_iterations = summary["iterations"]
            runtime = summary["runtime"]
            
            # Append the data to the table
            table_data.append([model_name, num_elements, valence_avg, valence_std, trust_region_iterations, runtime])

# Sort the table data by the number of elements
table_data.sort(key=lambda x: x[1])

# Generate LaTeX document
latex_document = r"""
\documentclass{article}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{booktabs}

\geometry{a4paper, margin=1in}

\begin{document}

\title{Optimization Log Summary}
\author{}
\date{}

\maketitle

\begin{table}[h!]
\centering
\begin{tabular}{|l|c|c|c|c|c|}
\hline
Model Name & Number of Elements & Valence Avg & Valence Std & Trust Region Iterations & Runtime (s) \\
\hline
"""

for row in table_data:
    latex_document += f"{row[0]} & {row[1]} & {row[2]:.2f} & {row[3]:.2f} & {row[4]} & {row[5]:.2f} \\ " + "\n" + r"\hline" + "\n"

latex_document += r"""
\end{tabular}
\caption{Optimization Log Summary}
\label{table:log_summary}
\end{table}

\end{document}
"""

# Save the LaTeX document to a file
output_path = os.path.join(os.path.dirname(__file__), 'log_summary_table.tex')
with open(output_path, 'w') as file:
    file.write(latex_document)

print("LaTeX document generated and saved to log_summary_table.tex")