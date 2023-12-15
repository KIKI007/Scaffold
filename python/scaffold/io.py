from scaffold.geometry import ScaffoldModel, StickModel
import os
import json
from scaffold import DATA_DIR
import numpy as np

class StickModelInput:
    def __init__(self):
        self.stick_model = StickModel()
        self.opt_parameters = {}

    def loadFile(self, file_name):
        file_path = os.path.join(DATA_DIR, file_name)
        with open(file_path, "r") as file:
            json_data = json.load(file)
            self.fromJSON(json_data)

    def saveFile(self, file_name="tmp.json"):
        file_path = os.path.join(DATA_DIR, file_name)
        print("Save file {} to the folder {}".format(file_name, DATA_DIR))
        with open(file_path, "w") as file:
            json.dump(self.toJSON(), file)

    def toJSON(self):
        data = {}
        data["stick_model"] = self.stick_model.toJSON()
        data["mt_config"] = self.opt_parameters
        return data

    def fromJSON(self, json_data):
        self.stick_model = StickModel()
        self.stick_model.fromJSON(json_data["stick_model"])
        self.opt_parameters = json_data["mt_config"]
        return


    def update_parameters(self):
        self.opt_parameters["clamp_t_bnd"] = self.opt_parameters.get(
            "clamp_t_bnd", 0.1)

        self.opt_parameters["pos_devi"] = self.opt_parameters.get("pos_devi", 0.05)

        self.opt_parameters["orient_devi"] = self.opt_parameters.get("orient_devi", 0.0574532925)

        self.opt_parameters["time_out"] = self.opt_parameters.get("time_out", 300)

        self.opt_parameters["bar_collision_distance"] = self.opt_parameters.get("bar_collision_distance", 0.02)

        self.opt_parameters["clamp_collision_dist"] = self.opt_parameters.get("clamp_collision_dist", 0.024)

        self.opt_parameters["contactopt_trust_region_start"] = self.opt_parameters.get("contactopt_trust_region_start",
                                                                                       0.1)
        self.opt_parameters["bar_available_lengths"] = self.opt_parameters.get("bar_available_lengths", [1.0, 2.0])

    def load_from_file_legacy(self, name):
        file_path = os.path.join(DATA_DIR, name)
        with open(file_path) as file:
            json_data = json.load(file)
            for point in json_data["nodes"]:
                point_coord = point["point"]
                self.stick_model.lineV.append(point_coord)
            self.stick_model.lineV = np.array(self.stick_model.lineV)

            for element in json_data["elements"]:
                self.stick_model.lineE.append(element["end_node_inds"])
            self.stick_model.lineE = np.array(self.stick_model.lineE)

            self.stick_model.file_name = name
            self.stick_model.radius = json_data.get('bar_radius', 0.01)
            self.opt_parameters = json_data.get('mt_config', {})
            if "normal_type" in self.opt_parameters\
                    and self.opt_parameters["normal_type"] == "Sphere":
                self.stick_model.computeSphereNormals()
                print(self.stick_model.normals)

class ScaffoldModelOutput:

    def __init__(self):
        self.scaffold_model = ScaffoldModel()
        self.opt_parameters = {}
        self.status = ""
        self.print_message = ""

    def toJson(self):
        data = {}
        if self.scaffold_model.is_valid():
            data["scaffold_model"] = self.scaffold_model.toJSON()

        data["opt_parameters"] = self.opt_parameters
        data["status"] = self.status
        data["print_message"] = self.print_message
        return data

    def fromJson(self, json_data):
        self.scaffold_model = ScaffoldModel()
        if "scaffold_model" in json_data:
            self.scaffold_model.fromJSON(json_data["scaffold_model"])

        self.opt_parameters = json_data["opt_parameters"]
        self.status = json_data["status"]
        self.print_message = json_data["print_message"]