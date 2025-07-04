import rhinoscriptsyntax as rs
from scaffold.io import StickModelInput
from scaffold import DATA_DIR
from os import listdir
from os.path import isfile, join
from pathlib import Path

from scaffold.io import StickModelInput
import numpy
import rhinocode
from scaffold.compute import computation_process, scaffold_optimization_gui_rhino
from multiprocessing import Process, Queue
import multiprocessing


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

option = rs.MessageBox("Load your own model?", 4 | 32)
stick_model_input = None
if option == 6:
    # input from rhino
    keys = ["Name", "Bar Radius (m)", "Bar Lengths (m)", "Clamp Gap (m)", "Position (m)", "Orientation (rad)",
            "Time (s)",
            "Debug"]
    values = ["scaffold", 0.01, [2.0], 0.016, 0.04, 0.087, 100, False]
    parameters = rs.PropertyListBox(keys, values, "Optimization Parameters")
    if parameters:
        filename = parameters[0]
        radius = float(parameters[1])
        bar_available_lengths = eval(parameters[2])
        clamp_gap = float(parameters[3])
        pos_devi = float(parameters[4])
        orient_devi = float(parameters[5])
        time_out = int(parameters[6])
        debug_mode = bool(parameters[7])

        rs.CurrentLayer("Default")
        input_curves = rs.GetObjects(message="input lines", filter=rs.filter.curve)
        lines = []
        for input_curve in input_curves:
            exploded_curves = rs.ExplodeCurves(input_curve, True)
            lines.extend(exploded_curves)
        fixed_lines = rs.GetObjects(message="fixed lines", filter=rs.filter.curve)

        if input_curves and fixed_lines:

            # parameters
            stick_model = {"radius": radius,
                           "vertices": [],
                           "edges": [],
                           "file_name": filename,
                           }

            parameters = {
                "fixed_element_ids": [],
                "layers": [],
                "reciprocal": False,
                "bar_available_lengths": bar_available_lengths,
                'clamp_gap': clamp_gap,
                'clamp_collision_dist': 2 * radius,
                'pos_devi': pos_devi,
                'orient_devi': orient_devi,
                'debug_mode': debug_mode,
                'time_out': time_out
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

            stick_model_input = StickModelInput()
            stick_model_input.fromJSON(data)
else:
    json_files = []
    for f in listdir(DATA_DIR):
        if isfile(join(DATA_DIR, f)):
            if Path(f).suffix == ".json":
                json_files.append(Path(f).stem)

    parameters = rs.MultiListBox(json_files, "Model Names")
    if parameters:
        filename = f"{parameters[0]}.json"
        stick_model_input = StickModelInput()
        stick_model_input.loadFile(filename)
        stick_model = stick_model_input.stick_model
        rs.CurrentLayer("Default")
        if rs.IsLayer("Input"):
            rs.PurgeLayer("Input")
        rs.AddLayer("Input")
        rs.CurrentLayer("Input")
        nodes = []

        for [ind0, ind1] in stick_model.lineE:
            np0 = stick_model.lineV[ind0, :]
            np1 = stick_model.lineV[ind1, :]
            rs.AddLine(np0.tolist(), np1.tolist())

if stick_model_input is not None:
    multiprocessing.set_executable(rhinocode.get_python_executable())
    print("Starting computation...")

    compute_queue = Queue()
    draw_queue = Queue()
    rhino_queue = Queue()
    compute_sub_process = Process(target=computation_process, args=(compute_queue, draw_queue))
    compute_sub_process.start()
    scaffold_optimization_gui_rhino(stick_model_input, compute_queue, draw_queue, rhino_queue)
    compute_sub_process.terminate()

    model = rhino_queue.get(block=False)
    print("Get")
    rs.CurrentLayer("Default")
    if rs.IsLayer("Output"):
        rs.PurgeLayer("Output")
    rs.AddLayer("Output")
    rs.CurrentLayer("Output")
    nodes = []
    radius = stick_model_input.stick_model.radius
    for [p0, p1] in model.lines:
        np0 = numpy.array(p0)
        np1 = numpy.array(p1)
        length = numpy.linalg.norm(np1 - np0)
        origin = p0
        normal = (np1 - np0) / length
        base_plane = rs.PlaneFromNormal(origin.tolist(), normal.tolist())
        rs.AddCylinder(base=base_plane, height=length, radius=radius)

    for id in range(len(model.adj)):
        for jd in range(0, 2):
            [V, F] = model.coupler(id, jd)
            rs.AddMesh(V.tolist(), F.tolist())