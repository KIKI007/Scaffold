from scaffold import DATA_DIR
from scaffold.geometry import StickModel, ScaffoldModel
import numpy as np
import json
import os

class ScaffoldModelImporter:

    def __init__(self):
        self.clear()

    def clear(self):
        self.lines = []
        self.adj = []
        self.cpts = []
        self.radius = []

    def parse(self, json_data):
        for point in json_data["line_pt_pairs"]:
            p0 = point[0]
            p1 = point[1]
            self.lines.append([p0, p1])

        for edge in json_data["contact_id_pairs"]:
            self.adj.append(edge)

        for cpts in json_data["coupler_pt_pairs"]:
            self.cpts.append(cpts)

        self.lines = np.array(self.lines)
        self.adj = np.array(self.adj)
        self.cpts = np.array(self.cpts)
        self.radius = json_data.get("radius", 0.01)

    @property
    def model(self):
        model = ScaffoldModel()
        model.load_geometry(self.lines, self.adj, self.cpts, self.radius)
        return model

    def read(self, filename):
        file_path = os.path.join(DATA_DIR, filename)

        self.clear()
        with open(file_path) as file:
            json_data = json.load(file)
            self.parse(json_data)

class StickModelImporter:

    def __init__(self):
        self.clear()

    def clear(self):
        self.lineV = []
        self.lineE = []
        self.parameters = []
        self.fixed_edge_ids = []
        self.bar_radius = 0

    @property
    def model(self):
        model = StickModel(self.lineV, self.lineE, self.bar_radius)
        return model

    @property
    def start_layer_id(self):
        return self.parameters.get("layer_range", [-1, -1])[0]

    @property
    def end_layer_id(self):
        end_layer_id =  self.parameters.get("layer_range", [-1, -1])[1]
        if end_layer_id >= 0:
            return end_layer_id
        else:
            return len(self.layers)

    @property
    def layers(self):
        return self.parameters.get('layers', [list(range(len(self.lineE)))])

    @property
    def debug_mode(self):
        return self.parameters.get('debug_mode', 0)

    @property
    def bar_available_lengths(self):
        return self.parameters.get('bar_available_lengths', [1])

    def parse(self, json_data):
        self.parse_geometry(json_data)
        self.parse_parameters(json_data)

    def parse_geometry(self, json_data):
        for point in json_data["nodes"]:
            point_coord = point["point"]
            self.lineV.append(point_coord)
        for element in json_data["elements"]:
            self.lineE.append(element["end_node_inds"])

        self.lineV = np.array(self.lineV)
        self.lineE = np.array(self.lineE)

    def parse_parameters(self, json_data):
        self.parameters = json_data['mt_config']
        self.fixed_edge_ids = json_data.get('fixed_element_ids', [])
        self.bar_radius = json_data.get('bar_radius', json_data["cross_secs"][0]["radius"])

        # sort bar_available_lengths
        bar_available_lengths = self.parameters.get("bar_available_lengths", [1])
        bar_available_lengths.sort()
        self.parameters["bar_available_lengths"] = self.bar_available_lengths

    def read(self, filename):
        file_path = os.path.join(DATA_DIR, filename)

        self.clear()
        with open(file_path) as file:
            json_data = json.load(file)
            self.parse(json_data)
