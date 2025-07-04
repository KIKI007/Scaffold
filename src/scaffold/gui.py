import numpy as np
import polyscope as ps
from scaffold import DATA_DIR
import polyscope.imgui as psim
from scaffold.io import StickModelInput, ScaffoldModelOutput
from scaffold.formfind.optimizer import SMILP_optimizer
from scaffold.collision import CollisionSolver
import time
import json
import os
from scaffold import LOCAL_SERVER_NAME

class ScaffoldViewer:
    def __init__(self):
        try:
            ps.init()
            ps.set_navigation_style("turntable")
            ps.set_up_dir("z_up")
            ps.set_ground_plane_mode('none')
            ps.set_user_callback(self.interface_scaffold)
            ps.remove_all_structures()
        except:
            pass

        self.show_clamps = False
        self.re_render = False

        self.model_select_table = ["None"]
        self.user_select_model_id = self.model_select_table[0]
        self.model_name_map = {}

        self.input = StickModelInput()
        self.models = []

        self.compute_queue = None
        self.draw_queue = None
        self.rhino_queue = None
        self.rhino = False

    def register_model(self, model):
        if hasattr(model, 'name'):
            self.model = model
            if model.name == "stick":
                self.register_lines_for_stickmodel(model)
            elif model.name == "scaffold":
                self.register_lines(model.lines, model.radius)
                if model.coupler_geometry != None:
                    self.register_couplers(model)

    def register_lines_for_stickmodel(self, model):
        ps.remove_curve_network("normal", False)

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
        coupler_id = 0
        while True:
            if ps.has_surface_mesh("coupler {}".format(coupler_id)):
                ps.remove_surface_mesh("coupler {}".format(coupler_id))
                coupler_id = coupler_id + 1
            else:
                break
        ps.remove_group("coupler", False)

        coupler_group = ps.create_group("coupler")
        coupler_group.set_show_child_details(False)
        coupler_group.set_hide_descendants_from_structure_lists(True)

        if self.show_clamps:
            for id in range(len(model.adj)):
                for jd in range(0, 2):
                    [V, F] = model.coupler(id, jd)
                    coupler_obj = ps.register_surface_mesh(name="coupler {}".format(2 * id + jd), vertices=V, faces=F, color=(1,1,0.1,1))
                    coupler_obj.add_to_group(coupler_group)

    def interface_scaffold(self):
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
        self.re_render = self.re_render or changed

        if self.get_current_model() != None:
            changed, self.get_current_model().radius= psim.InputFloat("radius", self.get_current_model().radius)
            self.re_render = self.re_render or changed

        changed, self.show_clamps = psim.Checkbox("Clamps", self.show_clamps)
        self.re_render = self.re_render or changed

        psim.SameLine()

        if psim.Button("Clear"):
            self.models = []
            self.model_select_table = ["None"]
            self.model_name_map = {}
            self.re_render = True

        psim.SameLine()
        if psim.Button("Collision"):
            model = self.get_current_model()
            if model != None:

                if self.show_clamps:
                    model.load_default_collision_coupler()
                else:
                    model.coupler_colliders = []

                solver = CollisionSolver(model)
                solver.solve()

        psim.PopItemWidth()
        self.update_render()

    def add_scaffold_model(self, model):
        self.models.append(model)
        self.model_select_table.append(model.name)
        self.model_name_map[model.name] = self.models[-1]
        self.user_select_model_id = model.name
        self.re_render = True

    def get_current_model(self):
        if self.user_select_model_id in self.model_name_map:
            return self.model_name_map[self.user_select_model_id]
        return None

    def show(self):
        ps.show()

    def update_render(self):
        if self.re_render:
            #ps.remove_all_structures()
            if hasattr(self, "input") and self.input.stick_model.is_valid():
                self.register_model(self.input.stick_model)
            if self.user_select_model_id != "None" and self.user_select_model_id in self.model_name_map:
                self.register_model(self.model_name_map[self.user_select_model_id])
            self.re_render = False
            #ps.reset_camera_to_home_view()

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

        ps.set_user_callback(self.interface_opt)

        # data
        self.running = False
        self.running_msg = ""

    def load_from_file(self, name):
        self.input = StickModelInput()
        self.input.loadFile(name)
        self.input.update_parameters()
        self.register_model(self.input.stick_model)

    def send_optimization_command(self):
        if self.input.stick_model.is_valid() and self.compute_queue is not None:
            msg = self.input.toJSON()
            self.input.saveFile()
            self.compute_queue.put(msg)
            self.running = True
            self.running_msg = ""

    def interface_opt(self):
        psim.PushItemWidth(150)
        # scaffold interface
        self.interface_scaffold()

        psim.Separator()
        # optimization button
        if not self.running:
            if psim.Button("Optimize"):
                self.send_optimization_command()

            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode("Parameters"):
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
        else:
            if self.draw_queue is not None:
                try:
                    draw_request = self.draw_queue.get(block=False)
                    output = ScaffoldModelOutput()
                    output.fromJson(draw_request)
                    self.running_msg = output.print_message
                    if output.status == "succeed" or output.status == "failed":
                        self.input.opt_parameters = output.opt_parameters
                        self.running = False
                        if self.rhino:
                            if len(self.models) > 0 and self.rhino_queue is not None:
                                self.rhino_queue.put(self.models[-1])
                            ps.unshow()
                        else:
                            ps.warning("Optimization finished.")
                    self.add_scaffold_model(output.scaffold_model)
                except:
                    pass

        self.update_render()