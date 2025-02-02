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

    def saveFile(self):
        file_path = os.path.join(DATA_DIR, self.stick_model.file_name)
        print("Save file {} to the folder {}".format(self.stick_model.file_name, DATA_DIR))
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
        self.update_parameters()
        return

    def update_parameters(self):
        self.opt_parameters["clamp_t_bnd"] = self.opt_parameters.get("clamp_t_bnd", 0.1)

        self.opt_parameters["pos_devi"] = self.opt_parameters.get("pos_devi", 0.04)

        self.opt_parameters["clamp_gap"] = self.opt_parameters.get("clamp_gap", 0.016)

        self.opt_parameters["orient_devi"] = self.opt_parameters.get("orient_devi", 0.0874532925)

        self.opt_parameters["time_out"] = self.opt_parameters.get("time_out", 300)

        self.opt_parameters["bar_collision_distance"] = self.opt_parameters.get("bar_collision_distance", 0.02)

        self.opt_parameters["clamp_collision_dist"] = self.opt_parameters.get("clamp_collision_dist", 0.024)

        self.opt_parameters["contactopt_trust_region_start"] = self.opt_parameters.get("contactopt_trust_region_start",
                                                                                       0.1)
        self.opt_parameters["bar_available_lengths"] = self.opt_parameters.get("bar_available_lengths", [1.0, 2.0])


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
