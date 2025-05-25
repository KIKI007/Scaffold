import time
import ghpythonlib.components as gh
from compas_eve import Message
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
import rhinoscriptsyntax as rs
from scaffold.io import StickModelInput
from scaffold.formfind.optimizer import SMILP_optimizer
import numpy

def end_points(line):
    p0 = rs.CurveStartPoint(line)
    p1 = rs.CurveEndPoint(line)
    return [p0.X, p0.Y, p0.Z], [p1.X, p1.Y, p1.Z]

def distance(p, q):
    np = numpy.array(p)
    nq = numpy.array(q)
    return numpy.linalg.norm(np - nq)

def get_line_id(line, lines):
    p0, p1 = end_points(line)
    for id, l in enumerate(lines):
        q0, q1 = end_points(l)
        dist0 = distance(p0, q0) + distance(p1, q1)
        dist1 = distance(p1, q0) + distance(p0, q1)
        if dist0 < 1E-6 or dist1 < 1E-6:
            return id

if __name__ == "__main__":
    # input from rhino
    radius = 0.01
    input_curves = rs.GetObjects(message = "input lines", filter = rs.filter.curve)
    lines = []
    for input_curve in input_curves:
        exploded_curves = rs.ExplodeCurves(input_curve, True)
        lines.extend(exploded_curves)
    fixed_lines = rs.GetObjects(message = "fixed lines", filter = rs.filter.curve)

    # parameters
    stick_model = {"radius": radius,
    "vertices": [],
    "edges": [],
    "file_name": "scaffold",
    }

    parameters = {
    "fixed_element_ids": [],
    "layers": [],
    "reciprocal": False,
    "bar_available_lengths": [2.0],
    'clamp_gap': 0.016,
    'clamp_collision_dist':2 * radius,
    'pos_devi': 0.04,
    'orient_devi': 0.0872664626,
    'debug_mode': False,
    'time_out': 100
    }
    
    for line in lines:
        p0, p1 = end_points(line)
        nV = len(stick_model["vertices"])
        stick_model["vertices"].append([x for x in p0])
        stick_model["vertices"].append([x for x in p1])
        stick_model["edges"].append([nV, nV + 1])
    
    remain_inds = list(range(len(lines)))
    if remain_inds != []:
        parameters["layers"].append(remain_inds)
    
    for line in fixed_lines:
        ind = get_line_id(line, lines)
        parameters["fixed_element_ids"].append(ind)
    
    data = {
        "stick_model": stick_model,
        "mt_config": parameters
    }
    
    topic = Topic("/scaffold/stick_model/", Message)
    tx = MqttTransport(host="localhost")
    publisher = Publisher(topic, transport=tx)
    msg = Message(data)
    publisher.publish(msg)
    time.sleep(1)
