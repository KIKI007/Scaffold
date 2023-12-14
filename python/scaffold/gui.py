import numpy as np
import polyscope as ps
from scaffold import DATA_DIR
import polyscope.imgui as psim
from scaffold.io import StickModelInput
from scaffold.formfind.optimizer import SMILP_optimizer
from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
import time
import json
import os

class ScaffoldViewer:
    def __init__(self):
        ps.init()
        ps.set_navigation_style("turntable")
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode('none')
        self.show_clamps = False

    def register_model(self, model):
        if hasattr(model, 'name'):
            self.model = model
            if model.name == "stick":
                self.register_lines_for_stickmodel(model)
            elif model.name == "scaffold":
                self.register_lines(model.lines, model.radius)
                if model.coupler_geometry != None and self.show_clamps:
                    self.register_couplers(model)

    def register_lines_for_stickmodel(self, model):
        # lines
        curves = ps.register_curve_network(name = "stick_frame", nodes = model.lineV, edges = model.lineE, color=(0.8, 0.5, 0), transparency=0.2)
        curves.set_radius(model.radius, relative=False)

        # normal lines
        if model.has_normals():
            V, E = model.computeNormalLines()
            curves = ps.register_curve_network(name="normal", nodes=V, edges=E, color=(0.5, 0.8, 0.6),
                                           transparency=0.2)

    def register_lines(self, lines, radius):
        nodes = []
        for [p0, p1] in lines:
            nodes.append(p0)
            nodes.append(p1)
        nodes = np.array(nodes)
        edges = [[i * 2, i * 2 + 1] for i in range(0, int(len(nodes) / 2))]
        edges = np.array(edges)
        curves = ps.register_curve_network(name = "frame", nodes = nodes, edges = edges, color = (0.5,0.5,1,1))
        curves.set_radius(radius, relative=False)

    def register_couplers(self, model):
        for id in range(len(model.adj)):
            for jd in range(0, 2):
                [V, F] = model.coupler(id, jd)
                coupler_obj = ps.register_surface_mesh(name="coupler" + str(2 * id + jd), vertices=V, faces=F, color=(1,1,0.1,1))

    def show(self):
        ps.show()

class ScaffoldOptimizerViewer(ScaffoldViewer):

    def __init__(self):
        ScaffoldViewer.__init__(self)
        self.interface_important = [
            {"name" : "pos_devi", "label": "position deviation (meter)", "type": "float_input"},
            {"name": "orient_devi", "label": "orientation deviation (radian)", "type": "float_input"},
            {"name": "clamp_t_bnd", "label": "scaffold clamp deviation (meter)", "type": "float_input"},
            {"name": "bar_collision_distance", "label": "minimum distance between beams", "type": "float_input"},
            {"name": "clamp_collision_dist", "label": "minimum distance between clamps", "type": "float_input"},
            {"name": "contactopt_trust_region_start", "label": "initial truss region size (meter)", "type": "float_input"},
            {"name": "time_out", "label": "optimization time out (second)", "type": "int_input"}
        ]

        self.interface_system = [

        ]
        ps.set_user_callback(self.set_customized_interface)

        self.model_select_table = ["None"]
        self.user_select_model_id = self.model_select_table[0]
        self.model_name_map = {}

        self.input = StickModelInput()
        self.models = []

        # data
        self.total_changed = False
        self.refresh = False
        self.running = False
        self.running_msg = ""

    def load_from_file_legacy(self, name):
        self.input = StickModelInput()
        self.input.load_from_file_legacy(name)
        self.input.update_parameters()
        self.register_model(self.input.stick_model)

    def load_from_file(self, name):
        self.input = StickModelInput()
        self.input.loadFile(name)
        self.input.update_parameters()
        self.register_model(self.input.stick_model)

    def send_optimization_command(self):
        if self.input.stick_model.is_valid():
            topic = Topic("/opt/problem_request/", Message)
            tx = MqttTransport(host="localhost")
            publisher = Publisher(topic, transport=tx)
            msg = Message(self.input.toJSON())
            self.input.saveFile()
            publisher.publish(msg)
            self.running = True
            self.refresh = False
            self.running_msg = ""
            time.sleep(1)

    def set_customized_interface(self):
        psim.PushItemWidth(150)

        # title
        if self.input.stick_model.is_valid():
            psim.TextUnformatted("Model:\t {} {}".format(self.input.stick_model.file_name, self.running_msg))

        # model for rendering
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        changed = psim.BeginCombo("Optimized Models", self.user_select_model_id)
        if changed:
            for val in self.model_select_table:
                _, selected = psim.Selectable(val, self.user_select_model_id == val)
                if selected:
                    self.user_select_model_id = val
                    self.show_clamps = False
            psim.EndCombo()
        self.total_changed = self.total_changed or changed

        psim.SameLine()

        changed, self.show_clamps = psim.Checkbox("Clamps", self.show_clamps)
        self.total_changed = self.total_changed or changed

        psim.Separator()

        # optimization button
        if self.running == False:

            if psim.Button("Optimize"):
                self.send_optimization_command()

            psim.SameLine()

            if psim.Button("Clear"):
                self.models = []
                self.model_name_map = {}
                self.refresh = True
                self.total_changed = True

            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Parameters (Important)"):
                for item in self.interface_important:
                    if item["type"] == "float_input":
                        changed, self.input.opt_parameters[item["name"]] = psim.InputFloat(item["label"],
                                                                                     self.input.opt_parameters[item["name"]])
                    elif item["type"] == "float_int":
                        changed, self.input.opt_parameters[item["name"]] = psim.InputInt(item["label"],
                                                                                   self.input.opt_parameters[item["name"]])

                if psim.TreeNode("Available Beam Lengths"):
                    lengths = self.input.opt_parameters["bar_available_lengths"]
                    for id in range(len(lengths)):
                        changed, lengths[id] = psim.InputFloat("length {}".format(id), lengths[id])
                    if psim.Button("+"):
                        lengths.append(1)
                    psim.SameLine()
                    if psim.Button("-") and len(lengths) > 0:
                        lengths.pop()

                psim.TreePop()

            psim.TreePop()

            if psim.TreeNode("Parameters (System)"):

                for item in self.interface_system:
                    if item["type"] == "float_input":
                        changed, self.input.opt_parameters[item["name"]] = psim.InputFloat(item["label"],
                                                                                     self.input.opt_parameters[item["name"]])
                    elif item["type"] == "float_int":
                        changed, self.input.opt_parameters[item["name"]] = psim.InputInt(item["label"],
                                                                                   self.input.opt_parameters[item["name"]])
            psim.TreePop()

        # update tables
        if  self.refresh:
            self.total_changed = True
            self.model_select_table = ["None"]
            self.model_name_map = {}
            for id in range(len(self.models)):
                name = "model{}".format(id)
                self.model_select_table.append(name)
                self.model_name_map[name] = self.models[id]
            self.user_select_model_id = self.model_select_table[-1]

        if self.total_changed:
            ps.remove_all_structures()
            self.register_model(self.input.stick_model)
            if self.user_select_model_id != "None" and self.user_select_model_id in self.model_name_map:
                self.register_model(self.model_name_map[self.user_select_model_id])

        self.total_changed = False
        self.refresh = False
