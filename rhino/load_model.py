import rhinoscriptsyntax as rs
from scaffold.io import StickModelInput
import numpy
import rhinocode
from scaffold.compute import computation_process, scaffold_optimization_gui_rhino
from multiprocessing import Process, Queue
import multiprocessing

keys = ["Name"]
values = ["box2x2"]
parameters = rs.PropertyListBox(keys, values, "Load Model")
if parameters:
    filename = f"{parameters[0]}.json"

input = StickModelInput()
input.loadFile(filename)
stick_model = input.stick_model
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