import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scaffold.geometry import StickModel
from scaffold.formfind.optimizer import SMILP_optimizer


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
                self.register_lines_for_stickmodel(model.lineV, model.lineE, model.radius)
            elif model.name == "scaffold":
                self.register_lines(model.lines, model.radius)
                if model.coupler_geometry != None and self.show_clamps:
                    self.register_couplers(model)

    def register_lines_for_stickmodel(self, nodes, edges, radius):
        curves = ps.register_curve_network(name = "stick_frame", nodes = nodes, edges = edges, color=(0.8, 0.5, 0), transparency=0.2)
        curves.set_radius(radius, relative=False)

    def register_lines(self, lines, radius):
        nodes = []
        for [p0, p1] in lines:
            nodes.append(p0)
            nodes.append(p1)
        nodes = np.array(nodes)
        edges = [[i * 2, i * 2 + 1] for i in range(0, int(len(nodes) / 2))]
        edges = np.array(edges)
        curves = ps.register_curve_network(name = "frame", nodes = nodes, edges = edges)
        curves.set_radius(radius, relative=False)

    def register_couplers(self, model):
        for id in range(len(model.adj)):
            for jd in range(0, 2):
                [V, F] = model.coupler(id, jd)
                coupler_obj = ps.register_surface_mesh(name="coupler" + str(2 * id + jd), vertices=V, faces=F, color=(1,1,0.1,1))

    def show(self):
        ps.show()

class ScaffoldOptimizerViewer(ScaffoldViewer, SMILP_optimizer):

    def __init__(self, name):
        ScaffoldViewer.__init__(self)
        SMILP_optimizer.__init__(self, name)
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

    def set_customized_interface(self):
        psim.PushItemWidth(150)

        total_changed = False
        if psim.Button("Optimize"):
            self.solve()
            total_changed = True

        psim.SameLine()

        if psim.Button("Refresh"):
            total_changed = True

        if total_changed:
            self.model_select_table = ["None"]
            self.model_name_map = {}
            for id in range(len(self.models)):
                name = "model{}".format(id)
                self.model_select_table.append(name)
                self.model_name_map[name] = self.models[id]
            self.user_select_model_id = self.model_select_table[-1]

        psim.SameLine()

        changed, self.show_clamps = psim.Checkbox("Clamps", self.show_clamps)
        total_changed = total_changed or changed

        psim.Separator()
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

        if psim.TreeNode("Models"):
            changed = psim.BeginCombo("Optimized Models", self.user_select_model_id)
            if changed:
                for val in self.model_select_table:
                    _, selected = psim.Selectable(val, self.user_select_model_id == val)
                    if selected:
                        self.user_select_model_id = val
                        self.show_clamps = False
                psim.EndCombo()
            total_changed = total_changed or changed

        psim.TreePop()

        if total_changed:
            ps.remove_all_structures()
            self.register_model(self.stick_model)
            if self.user_select_model_id != "None" and self.user_select_model_id in self.model_name_map:
                self.register_model(self.model_name_map[self.user_select_model_id])


        psim.Separator()
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

        if psim.TreeNode("Parameters (Important)"):
            for item in self.interface_important:
                if item["type"] == "float_input":
                    changed,  self.opt_parameters[item["name"]] = psim.InputFloat(item["label"], self.opt_parameters[item["name"]])
                elif item["type"] == "float_int":
                    changed,  self.opt_parameters[item["name"]] = psim.InputInt(item["label"], self.opt_parameters[item["name"]])
        psim.TreePop()

        if psim.TreeNode("Parameters (System)"):

            for item in self.interface_system:
                if item["type"] == "float_input":
                    changed, self.opt_parameters[item["name"]] = psim.InputFloat(item["label"],
                                                                                 self.opt_parameters[item["name"]])
                elif item["type"] == "float_int":
                    changed, self.opt_parameters[item["name"]] = psim.InputInt(item["label"],
                                                                               self.opt_parameters[item["name"]])

