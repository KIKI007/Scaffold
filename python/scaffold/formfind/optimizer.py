import os
import json
import numpy as np
from scaffold import DATA_DIR, MT_DIR
from termcolor import cprint
from scaffold.formfind.util import *
from scaffold.formfind.post_processing import *
from scaffold.formfind.gurobi_func import *
from scaffold.io import ScaffoldModelOutput, StickModelInput

from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
import time
import multiprocessing as mp
from multiprocessing import Process


class SMILP_optimizer:

    # Difference between edge_coord and edge_ind
    # edges_coord: a list of two value vectors each of which represents an edge's two ends' vertex index
    # edges_ind: a list of int value, each item (i) corresponds to an edge (edge_coord[i])
    # due to legacy, this class (i/o) still utilizes both
    # however, for reset functions, we prefer edge_coord
    # as using layers, could strongly affect the edge_ind, leading to re-index each of its items, causing troubles
    # however, no matter how reshuffle, edges_coord still can represent the same edge

    def __init__(self, name):
        self.file_name = name
        pass

    def from_edge_ind_to_coord(self, sub_inds):
        return [self.line_edges_coord[eit] for eit in sub_inds]

    def collect_layer_edge_coords(self, layer_end):
        edge_coords = set()
        for fe in self.fixed_edges_coord:
            edge_coords.add(tuple(fe))

        for layer_id in range(0, layer_end + 1):
            for le in self.layers_coord[layer_id]:
                edge_coords.add(tuple(le))
        return [[e[0], e[1]] for e in edge_coords]

    def input_model(self, input):

        if "." not in self.file_name:
            self.file_name += ".json"

        self.line_vertices_coord = []
        self.line_edges_coord = []

        self.layer_start = -1
        self.layer_end = -1

        self.parse_from_input(input)
        self.update_optimization_parameters()
        self.update_geometry()

        self.check_geometry_match_barlengths()
        self.parse_prev_computed_result()

    def parse_from_input(self, input):
        self.bar_radius = input.stick_model.radius

        # load geometry
        self.line_vertices_coord = input.stick_model.lineV.copy()
        self.line_edges_coord = input.stick_model.lineE.copy()
        if input.stick_model.has_normals():
            self.normals = input.stick_model.normals.copy()

        self.stick_model = input.stick_model

        # parameters
        self.opt_parameters = input.opt_parameters.copy()

        # edges that are not optimized
        self.fixed_edges_ind = self.opt_parameters.get('fixed_element_ids', [])
        self.fixed_edges_coord = self.from_edge_ind_to_coord(self.fixed_edges_ind)
        del self.fixed_edges_ind

        # layers' index
        self.layers_ind = self.opt_parameters.get('layers', [list(range(len(self.line_edges_coord)))])

        # remove fixed edges' index
        self.layers_coord = []
        if len(self.layers_ind) == 0:
            single_layer_coord = self.line_edges_coord
            for ecoord in self.fixed_edges_coord:
                single_layer_coord.remove(ecoord)
            self.layers_coord = [single_layer_coord]
        else:
            for layer_ind in self.layers_ind:
                layer_coord = self.from_edge_ind_to_coord(layer_ind)
                for ecoord in self.fixed_edges_coord:
                    if ecoord in layer_coord:
                        layer_coord.remove(ecoord)
                self.layers_coord.append(layer_coord)
        del self.layers_ind

    def update_optimization_parameters(self):

        # only optimizes the layer between the layer_start and layer_end
        [self.layer_start, self.layer_end] = self.opt_parameters.get("layer_range", [-1, -1])
        self.layer_end = self.layer_end if self.layer_end >= 0 else len(self.layers_coord)

        # activate debug mode leads to detailed information for optimization
        self.debug_mode = self.opt_parameters.get('debug_mode', 0)

        # available bar lengths (list discrete numbers)
        self.beam_available_lengths = self.opt_parameters.get("bar_available_lengths", [1.0, 2.0])
        self.beam_available_lengths.sort()
        self.opt_parameters["bar_available_lengths"] = self.beam_available_lengths

        # the clamp thickness from both sides
        self.clamp_gap = self.opt_parameters.get("clamp_gap", 0.016)

        # half the true distance between two beams
        # the distance between two beams is radius plus the clamps thickness from one side
        self.bar_contact_distance = self.bar_radius + self.clamp_gap / 2

        self.opt_parameters.update({
            # * internal
            "opt_tol": 1E-2,
            "max_tangent": -1,
            "trust_region_lobnd": 1E-6,
            "trust_region_upbnd": 1.0,
            "contact_ignore_para_tol": 1E-3,
            "line_para_tol": 1E-2,
            "bigM": 1,
            "pos_tol": 1E-6,
            "duplicate_vertices_tol": 1E-3,
            "bar_contact_distance_tol": 0,
            "bar_contact_distance": self.bar_contact_distance,
            "fixed_contact_pairs_coord": [],
            "clamp_extension_length": 0.03,
            "debug_mode": self.debug_mode
        })

        self.opt_parameters["num_unsuccessful_step_to_expand_trust_region"] = self.opt_parameters.get("num_unsuccessful_step_to_expand_trust_region", 10)
        self.opt_parameters["clamp_t_bnd"] = self.opt_parameters.get("clamp_t_bnd", 0.1)
        self.opt_parameters["pos_devi"] = self.opt_parameters.get("pos_devi", 0.05)
        self.opt_parameters["orient_devi"] = self.opt_parameters.get("orient_devi", 0.0574532925)
        self.opt_parameters["time_out"] = self.opt_parameters.get("time_out", 300)
        self.opt_parameters["bar_collision_distance"] = self.opt_parameters.get("bar_collision_distance", 0.02)
        self.opt_parameters["clamp_collision_dist"] = self.opt_parameters.get("clamp_collision_dist", 0.024)
        self.opt_parameters["contactopt_trust_region_start"] = self.opt_parameters.get("contactopt_trust_region_start",
                                                                                       0.1)
        self.opt_parameters["reciprocal"] = self.opt_parameters.get("reciprocal", False)

        self.prev_opt_data = {"vs": [], "xs": [], "nstatus": -1, "contact_pairs_coord": []}

    def send_result_message(self, queue):
        data = queue.get()
        topic = Topic("/opt/scaffold_model/", Message)
        tx = MqttTransport(host="localhost")
        publisher = Publisher(topic, transport=tx)
        msg = Message(data)
        publisher.publish(msg)
        time.sleep(1)

    def send_result(self, status="", print_message="", model=None):
        output = ScaffoldModelOutput()
        if model is not None:
            output.scaffold_model = model
        output.opt_parameters = self.opt_parameters
        output.status = status
        output.print_message = print_message

        data = output.toJson()

        queue = mp.Queue()
        p = Process(target=self.send_result_message, args=(queue,))
        p.start()
        queue.put(data)
        p.join()

    def parse_prev_computed_result(self):
        self.models = []
        if self.layer_start >= 0:
            # read
            mt_file = os.path.join(MT_DIR,
                                   self.file_name.split(".")[0]
                                   + "_MT" + "_layer_"
                                   + str(self.layer_start) + ".json")

            if os.path.isfile(mt_file):
                with open(mt_file) as file:

                    json_data = json.load(file)
                    xs = json_data["opt_line_pos"]
                    xs = [np.array(x) for x in xs]

                    vs = json_data["opt_line_orient"]
                    vs = [np.array(v) for v in vs]

                    contact_pairs_coord = []

                    if "contact_pairs_coord" in json_data:
                        contact_pairs_coord = json_data["contact_pairs_coord"]

                    elif "contact_id_pairs" in json_data:
                        # legacy code
                        contact_id_pairs = json_data["contact_id_pairs"]
                        contact_pairs_coord = [(self.line_edges_coord[e0], self.line_edges_coord[e1]) for [e0, e1] in
                                               contact_id_pairs]

                    self.opt_parameters["fixed_contact_pairs_coord"] = contact_pairs_coord

                    opt_data = {"vs": vs,
                                "xs": xs,
                                "bar_radius": self.bar_radius,
                                "contact_pairs_coord": contact_pairs_coord,
                                "fixed_edges_coord": self.fixed_edges_coord,
                                "vertices_coord": self.line_vertices_coord.copy(),
                                "nstatus": 0}

                    opt_data["edges_coord"] = json_data.get("opt_edges_coord",
                                                            self.collect_layer_edge_coords(self.layer_start))

                    model = compute_scaffold_model(self.center_before_update, opt_data, self.opt_parameters, False)
                    self.send_result("loading", "loading", model)

                    model = ScaffoldModel()
                    model.fromJSON(self.models[-1].toJSON())
                    self.models.append(model)

                    self.prev_opt_data = opt_data.copy()
            else:
                self.layer_start = -1

    def update_geometry(self):

        # * center the structure
        self.center_before_update = np.array([0, 0, 0])
        for v in self.line_vertices_coord:
            self.center_before_update = self.center_before_update + v
        self.center_before_update /= len(self.line_vertices_coord)

        for v in self.line_vertices_coord:
            v -= self.center_before_update

        # remove duplication
        # * preprocessing to remove duplicate points and resolve joints in the middle of bars

        if self.stick_model.has_normals():
            [self.line_vertices_coord,
             self.line_edges_coord,
             self.fixed_edges_coord,
             self.layers_coord,
             self.normals] = remove_vertex_duplication(self.line_vertices_coord,
                                                                   self.line_edges_coord,
                                                                   self.fixed_edges_coord,
                                                                   self.layers_coord,
                                                                   self.opt_parameters,
                                                                   self.normals)
        else:
            [self.line_vertices_coord,
             self.line_edges_coord,
             self.fixed_edges_coord,
             self.layers_coord] = remove_vertex_duplication(self.line_vertices_coord,
                                                            self.line_edges_coord,
                                                            self.fixed_edges_coord,
                                                            self.layers_coord,
                                                            self.opt_parameters)

        if self.opt_parameters["reciprocal"] and self.stick_model.has_normals():
            self.opt_parameters["fixed_contact_pairs_coord"], self.opt_parameters["fixed_contact_pairs_normals"] = compute_reciprocal_contact_pairs(self.line_vertices_coord, self.line_edges_coord, self.normals)

    def check_geometry_match_barlengths(self):
        # * check if any bar is too long
        max_available_length = max(self.beam_available_lengths)
        for i in range(len(self.line_edges_coord)):
            length = np.linalg.norm(self.line_vertices_coord[self.line_edges_coord[i][0]] - self.line_vertices_coord[
                self.line_edges_coord[i][1]])
            if length > max_available_length:
                raise ValueError(
                    "Line #{} has length {:.3f}, exceeding maximum available length {}. Please scale down your design.".format(
                        i, length, max_available_length))

    def solve(self):

        self.models = []
        self.parse_prev_computed_result()

        curr_edges_coord = self.fixed_edges_coord.copy()

        curr_fixed_edges_coord = self.fixed_edges_coord.copy()

        opt_data = self.prev_opt_data.copy()

        for curr_layer_id, curr_layer in enumerate(self.layers_coord):
            if curr_layer_id > self.layer_end:
                break

            # collect edges for each layer
            cprint('Solving layer {}/{}'.format(curr_layer_id, len(self.layers_coord) - 1), "yellow")

            curr_edges_coord.extend(curr_layer)

            # skip layers that has been computed
            if curr_layer_id <= self.layer_start:
                curr_fixed_edges_coord = curr_edges_coord.copy()
                continue

            # run optimization
            opt_data = self.run_opt(curr_edges_coord,
                                    self.line_vertices_coord,
                                    curr_fixed_edges_coord,
                                    xs=opt_data["xs"],
                                    vs=opt_data["vs"])

            if opt_data["nstatus"] != 0:
                cprint('No solution found!', 'red')
                self.send_result("failed", "no solution")
                return False

            curr_fixed_edges_coord = curr_edges_coord.copy()

            opt_data.update(
                {
                    "vertices_coord": self.line_vertices_coord,
                    "edges_coord": curr_edges_coord,
                    "fixed_edges_coord": self.fixed_edges_coord,
                    "bar_radius": self.bar_radius,
                    "nstatus": 0
                }
            )

            self.opt_parameters["fixed_contact_pairs_coord"] = opt_data["contact_pairs_coord"]

            running_message = "running (layer_id {})".format(curr_layer_id)

            model = compute_scaffold_model(self.center_before_update,
                                           opt_data,
                                           self.opt_parameters,
                                           True,
                                           {"name": self.file_name, "complete": curr_layer_id + 1 == self.layer_end,
                                            "id": curr_layer_id})

            self.send_result("running", running_message, model)

        if opt_data["nstatus"] != 0:
            cprint('No solution found!', 'red')
            self.send_result("failed", "no solution")
            return False

        self.send_result("succeed", "find solution", model)
        return True

    def run_opt(self, E, V, FE, vs=None, xs=None):

        # load parameters
        contactopt_trust_region_size_start = self.opt_parameters["contactopt_trust_region_start"]
        contactopt_trust_region_size_upbnd = self.opt_parameters["trust_region_upbnd"]
        contactopt_trust_region_size_lobnd = self.opt_parameters["trust_region_lobnd"]
        optimality_tol = self.opt_parameters["opt_tol"]
        bar_distance = self.opt_parameters["bar_contact_distance"]
        clamp_distance = self.opt_parameters["clamp_collision_dist"]
        num_step_to_expand = self.opt_parameters["num_unsuccessful_step_to_expand_trust_region"]

        # initialization
        vs = vs or []
        xs = xs or []
        num_it = 0

        # contact assignment
        # prev_radius = 0
        # prev_collision_dist = 0
        tr_size = contactopt_trust_region_size_start
        num_same_tr_it = 0
        while tr_size > contactopt_trust_region_size_lobnd and tr_size < contactopt_trust_region_size_upbnd:
            num_it = num_it + 1
            result = run_beamopt(E, V, FE, vs, xs, tr_size, self.opt_parameters)
            if result != None:
                [vs, xs, curr_radius, curr_collision_dist, contact_pairs] = result

                cprint("it = {}, tr_size = {:.2e}, radius = {:.2e}, clamp_dist = {:.2e}".format(num_it, tr_size,
                                                                                                curr_radius,
                                                                                                curr_collision_dist),
                       'cyan')

                opt_data = {"vs": vs, "xs": xs,
                            "contact_pairs_coord": contact_pairs,
                            "nstatus": -1,
                            "bar_radius": self.bar_radius,
                            "edges_coord": E,
                            "fixed_edges_coord": self.fixed_edges_coord,
                            "vertices_coord": self.line_vertices_coord.copy()}

                model = compute_scaffold_model(self.center_before_update, opt_data, self.opt_parameters, False)

                self.send_result("opt", "(it={}, tr_size={:.2e}, radius={:.2e})".format(num_it, tr_size, curr_radius),
                                 model)

                if curr_radius >= (1 - optimality_tol) * bar_distance \
                        and curr_collision_dist >= (1 - optimality_tol) * clamp_distance:
                    tr_size /= 2
                    num_same_tr_it = 0

                if num_same_tr_it >= num_step_to_expand:
                    tr_size *= 2
                    num_same_tr_it = 0

                num_same_tr_it = num_same_tr_it + 1
            else:
                cprint("it = {}, tr_size = {}. Failed".format(num_it, tr_size), 'red')
                tr_size *= 2
                num_same_tr_it = 0

        # evaluate result
        if tr_size >= contactopt_trust_region_size_upbnd:
            nstatus = -1
            contact_pairs = []
        else:
            nstatus = 0

        # save optimization
        opt_data = {"vs": vs, "xs": xs, "contact_pairs_coord": contact_pairs, "nstatus": nstatus}
        return opt_data