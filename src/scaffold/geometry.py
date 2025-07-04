import numpy as np
from scaffold import COUPLER_OBJ_PATH, COUPLER_COARSE_OBJ_PATH, COUPLER_COLLI_OBJ_PATHs
import xml.etree.cElementTree as ET
import json
from scaffold import DATA_DIR
import os
import numba
import trimesh

class StickModel:

    def load_geometry(self, lineV, lineE, radius):
        self.lineV = np.copy(lineV)
        self.lineE = np.copy(lineE)
        self.radius = radius

    def __init__(self):
        self.name = "stick"
        self.file_name = ""
        self.lineV = np.array([])
        self.lineE = np.array([])
        self.radius = 0

    def is_valid(self):
        if self.lineV.shape[0] != 0 and self.lineE.shape[0] != 0:
            return True
        else:
            return False

    def has_normals(self):
        if hasattr(self, "normals"):
            return True
        else:
            return False

    def toJSON(self):
        data = {
            "vertices": self.lineV.tolist(),
            "edges": self.lineE.tolist(),
            "radius": self.radius,
            "file_name": self.file_name
        }

        if self.has_normals():
            data["normals"] = self.normals.tolist()

        return data

    def fromJSON(self, json_data):
        self.lineV = np.array(json_data["vertices"])
        self.lineE = np.array(json_data["edges"])
        self.radius = json_data["radius"]
        if "normals" in json_data:
            self.normals = np.array(json_data["normals"])
        self.file_name = json_data["file_name"]

    def computeNormalLines(self, length = 0.1):
        V = []
        E = []
        for id, v in enumerate(self.lineV):
            V.append(v)
            V.append(v + self.normals[id] * length)
            E.append([len(V) - 2, len(V) - 1])
        V = np.array(V)
        E = np.array(E)
        return V, E

    def computeSphereNormals(self):
        self.normals = []
        for v in self.lineV:
            self.normals.append(v / np.linalg.norm(v))
        self.normals = np.array(self.normals)

class ScaffoldModel:

    def load_geometry(self, lines, adjacency, contact_points, radius):
        # data
        self.lines = np.copy(lines)
        self.adj = np.copy(adjacency)
        self.coupler_signs = None
        self.coupler_contact_pts = np.copy(contact_points)
        self.radius = radius
        self.compute_center()

    def __init__(self, model = None):
        if model is None:
            self.name = "scaffold"
            # geometry asset
            self.coupler_geometry = None
            self.coupler_colliders = []
            self.coupler_collider_names = []
        else:
            self.name = model.name
            self.coupler_geometry = model.coupler_geometry
            self.coupler_colliders = model.coupler_colliders
            self.coupler_collider_names = model.coupler_collider_names
            self.lines = model.lines
            self.adj = model.adj
            self.coupler_signs = model.coupler_signs
            self.coupler_contact_pts = model.coupler_contact_pts
            self.radius = model.radius
            self.compute_center()

    def is_valid(self):
        if hasattr(self, "lines") and hasattr(self, "adj") and hasattr(self, "coupler_contact_pts") and hasattr(self, "radius"):
            return True
        else:
            return False

    def toJSON(self):
        data = {
            "line": self.lines.tolist(),
            "adj": self.adj.tolist(),
            "coupler_contact_pts": self.coupler_contact_pts.tolist(),
            "radius": self.radius,
            "half_couplers": self.halfCouplersJson()
        }

        return data

    def halfCouplersJson(self):
        half_couplers = []
        for coupler_id in range(len(self.adj)):
            for side_id in range(2):
                half_couplers_data = {}
                half_couplers_data["pose"] = []

                mat, vec = self.coupler_transformation_matrix(coupler_id, side_id)
                mat = mat.T
                T = np.eye(4, dtype=float)
                T[:3, :3] = mat
                T[:3, 3]  = vec
                for id in range(4):
                    half_couplers_data["pose"].append(T[:, id].tolist())
                half_couplers_data["at_element"] = int(self.adj[coupler_id][side_id])
                half_couplers_data["to_element"] = int(self.adj[coupler_id][(side_id + 1) % 2])
                half_couplers.append(half_couplers_data)
        return half_couplers

    def fromJSON(self, json_data):
        self.name = "scaffold"
        self.lines = np.array(json_data["line"])
        self.adj = np.array(json_data["adj"])
        self.coupler_contact_pts = np.array(json_data["coupler_contact_pts"])
        self.radius = json_data["radius"]
        self.coupler_signs = None
        self.compute_center()
        self.load_default_coupler()

    def compute_center(self):
        self.center = np.array([0, 0, 0], dtype=float)
        for id in range(len(self.lines)):
            psta = np.array(self.lines[id][0])
            pend = np.array(self.lines[id][1])
            self.center += (psta + pend) / 2;
        self.center /= len(self.lines)

    def beam_mat4(self, line_id):
        height = np.linalg.norm(self.lines[line_id][1] - self.lines[line_id][0])

        # axis
        zaxis = self.lines[line_id][1] - self.lines[line_id][0]
        zaxis /= np.linalg.norm(zaxis)

        # yaxis
        yaxis = np.cross(np.array([0, 0, 1]), zaxis)
        if np.linalg.norm(yaxis) < 1E-6:
            yaxis = np.array([0, 1, 0]).cross(zaxis)
        yaxis /= np.linalg.norm(yaxis)

        # xaixs
        xaxis = np.cross(yaxis, zaxis)
        xaxis /= np.linalg.norm(xaxis)

        # orgin
        origin = (self.lines[line_id][1] + self.lines[line_id][0]) / 2

        # mat4
        xaxis = np.vstack([*xaxis, 0])
        yaxis = np.vstack([*yaxis, 0])
        zaxis = np.vstack([*zaxis, 0])
        origin = np.vstack([*origin, 1])
        mat = np.hstack([xaxis, yaxis, zaxis, origin])
        return mat, self.radius, height

    def coupler_transformation_matrix(self, coupler_id, side_id):
        edge_id = self.adj[coupler_id][side_id]
        zaxis = self.lines[edge_id][1] - self.lines[edge_id][0]
        zaxis /= np.linalg.norm(zaxis)
        if self.coupler_signs != None:
            zaxis *= self.coupler_signs[coupler_id][side_id]

        other_side_id = (side_id + 1) % 2
        xaxis = self.coupler_contact_pts[coupler_id][side_id] - self.coupler_contact_pts[coupler_id][other_side_id]
        xaxis /= np.linalg.norm(xaxis)
        yaxis = np.cross(zaxis, xaxis)
        mat = np.vstack([xaxis, yaxis, zaxis])
        vec = self.coupler_contact_pts[coupler_id][side_id]
        return mat, vec

    def coupler_mat4(self, coupler_id, side_id):
        mat, vec = self.coupler_transformation_matrix(coupler_id, side_id)
        mat4 = np.eye(4)
        mat4[:3, :3] = mat.T
        mat4[:3, 3] = vec
        return mat4

    def coupler(self, coupler_id, side_id):
        if self.coupler_geometry != None:
            mat, vec = self.coupler_transformation_matrix(coupler_id, side_id)
            V = np.copy(self.coupler_geometry["V"])
            V = V @ mat + np.tile(vec, (V.shape[0], 1))
            return [V, self.coupler_geometry["F"]]

    def load_default_coupler(self):
        mesh = trimesh.load_mesh(COUPLER_OBJ_PATH)
        self.coupler_geometry = {"V" : np.array(mesh.vertices), "F" : np.array(mesh.faces, dtype=np.int32)}

    def load_default_coupler_coarse(self):
        mesh = trimesh.load_mesh(COUPLER_COARSE_OBJ_PATH)
        self.coupler_geometry = {"V" : np.array(mesh.vertices), "F" : np.array(mesh.faces, dtype=np.int32)}

    def load_default_collision_coupler(self):
        self.coupler_colliders = []

        if abs(self.radius - 0.01) < 1E-6:
            for path in COUPLER_COLLI_OBJ_PATHs:
                mesh = trimesh.load_mesh(path)
                self.coupler_colliders.append([mesh.vertices.tolist(), mesh.faces.tolist()])

