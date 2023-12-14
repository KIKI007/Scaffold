import numpy as np
import igl
from scaffold import COUPLER_OBJ_PATH, COUPLER_COARSE_OBJ_PATH, COUPLER_COLLI_OBJ_PATHs
import xml.etree.cElementTree as ET
import json
from scaffold import DATA_DIR
import os

class StickModel:

    def load_geometry(self, lineV, lineE, radius):
        self.lineV = np.copy(lineV)
        self.lineE = np.copy(lineE)
        self.radius = radius

    def __init__(self):
        self.name = "stick"
        self.file_name = ""
        self.lineV = []
        self.lineE = []
        self.radius = 0

    def is_valid(self):
        if self.lineV is not [] and self.lineE is not []:
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

    def __init__(self, ):
        self.name = "scaffold"

        # geometry asset
        self.coupler_geometry = None
        self.coupler_colliders = []
        self.coupler_collider_names = []

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
            "radius": self.radius
        }
        return data

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

    def coupler(self, coupler_id, side_id):
        if self.coupler_geometry != None:
            mat, vec = self.coupler_transformation_matrix(coupler_id, side_id)
            V = np.copy(self.coupler_geometry["V"])
            V = V @ mat + np.tile(vec, (V.shape[0], 1))
            return [V, self.coupler_geometry["F"]]

    def load_default_coupler(self):
        V, F = igl.read_triangle_mesh(COUPLER_OBJ_PATH)
        self.coupler_geometry = {"V" : np.array(V), "F" : np.array(F)}

    def load_default_coupler_coarse(self):
        V, F = igl.read_triangle_mesh(COUPLER_COARSE_OBJ_PATH)
        self.coupler_geometry = {"V": np.array(V), "F": np.array(F)}

    def load_default_collision_coupler(self):
        V = []
        F = []
        for path in COUPLER_COLLI_OBJ_PATHs:
            pV, pF = igl.read_triangle_mesh(path)
            pV = np.array(pV)
            pF = np.array(pF)
            self.coupler_colliders.append([pV, pF])


    ##################
    ### for mujoco ###
    ##################

    def write_xml_asset(self, root):
        asset = ET.SubElement(root, "asset")
        id = 0
        for path in COUPLER_COLLI_OBJ_PATHs:
            name_str = "coupler{}".format(id)
            self.coupler_collider_names.append(name_str)
            ET.SubElement(asset, "mesh", name=name_str, file=path)
            id = id + 1

    def write_xml_beam(self, doc):
        for id in range(len(self.lines)):
            psta = np.array(self.lines[id][0]) - self.center
            pend = np.array(self.lines[id][1]) - self.center
            fromto_str = "{} {} {} {} {} {}".format(*psta, *pend)
            body = ET.SubElement(doc, "body")
            beam_name_str = "beam{}".format(id)
            ET.SubElement(body, "geom", name=beam_name_str, type="cylinder", size=str(self.radius), rgba="0.3 .3 0.9 1",
                          fromto=fromto_str)
            self.body_cylinder_names.append(beam_name_str)

    def write_xml_couplers(self, doc):
        coupler_id = 0
        for id in range(len(self.adj)):
            for jd in range(0, 2):
                mat, vec = self.coupler_transformation_matrix(id, jd)
                vec = vec - self.center
                xyaxes_str = "{} {} {} {} {} {}".format(mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1],
                                                        mat[1][2])
                pos_str = "{} {} {}".format(vec[0], vec[1], vec[2])
                body = ET.SubElement(doc, "body")
                for mesh_name in self.coupler_collider_names:
                    coupler_name = "coupler{}".format(coupler_id)
                    coupler_id = coupler_id + 1
                    ET.SubElement(body, "geom", name=coupler_name, type="mesh", mesh=mesh_name, xyaxes=xyaxes_str,
                                  pos=pos_str, rgba="1 1 0.1 1")
                    self.body_coupler_names.append(coupler_name)

    def write_xml_pair(self, root):
        contact = ET.SubElement(root, "contact")
        for coupler_name in self.body_cylinder_names:
            for cylinder_name in self.body_cylinder_names:
                ET.SubElement(contact, "pair", geom1=coupler_name, geom2=cylinder_name)

    def save_to_xml(self):
        root = ET.Element("mujoco")
        doc = ET.SubElement(root, "worldbody")
        ET.SubElement(doc, "light", diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")

        self.write_xml_asset(root)
        self.write_xml_beam(doc)
        self.write_xml_couplers(doc)
        self.write_xml_pair(root)

        tree = ET.ElementTree(root)
        tree.write("file.xml")
        return ET.tostring(root, encoding='us-ascii')

