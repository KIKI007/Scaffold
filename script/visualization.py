import json
from scaffold.gui import ScaffoldViewer
from scaffold.geometry import ScaffoldModel
from scaffold import MT_DIR
import os
import argparse

def scaffold_visualization_gui(file_path):
    viewer = ScaffoldViewer()
    model = ScaffoldModel()
    if file_path is None or file_path == "":
        file_path = "one_tet_layer_0.json"
    path = os.path.join(MT_DIR, file_path)
    with open(path) as file:
        json_assembly = json.load(file)
        model.fromJSON(json_assembly)
        model.load_default_collision_coupler()
        viewer.add_scaffold_model(model)
    viewer.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computational design and fabrication of reusable multi-tangent bar structures')
    parser.add_argument('--name', type=str, default = "box2x2", help='The name of result model to visualize (no extension)')

    args = parser.parse_args()
    file_path = args.name

    file_path = f"{file_path}_layer_0.json"
    scaffold_visualization_gui(file_path)