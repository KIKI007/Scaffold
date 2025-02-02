import json
from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
from scaffold.gui import ScaffoldOptimizerViewer, ScaffoldViewer
from scaffold.formfind.optimizer import SMILP_optimizer
from scaffold.io import StickModelInput, ScaffoldModelOutput
from scaffold.geometry import ScaffoldModel
from scaffold import MT_DIR
import multiprocessing as mp
from multiprocessing import Process, Queue
import time
import os
from scaffold import REMOTE_SERVER_NAME, LOCAL_SERVER_NAME

def update_viewer_stick_model(msg):
    global viewer
    if viewer.running == False:
        modelinput = StickModelInput()
        modelinput.fromJSON(msg.data)
        viewer.input = modelinput
        viewer.re_render = True

def update_viewer_scaffold_models(msg):
    global viewer
    if viewer.running == True:
        output = ScaffoldModelOutput()
        output.fromJson(msg.data)
        viewer.running_msg = output.print_message

        if output.status == "succeed" or output.status == "failed":
            viewer.input.opt_parameters = output.opt_parameters
            viewer.running = False
            return

        viewer.add_scaffold_model(output.scaffold_model)

def update_optimization(msg):
    input = StickModelInput()
    input.fromJSON(msg.data)
    optimizer = SMILP_optimizer(input.stick_model.file_name)
    optimizer.input_model(input)
    optimizer.solve()

def scaffold_visualization(file_path):
    global viewer
    viewer = ScaffoldViewer()
    model = ScaffoldModel()
    if file_path is None or file_path == "":
        file_path = "one_tet_layer_0.json"
    path = os.path.join(MT_DIR, file_path)
    with open(path) as file:
        json_assembly = json.load(file)
        model.fromJSON(json_assembly)
        model.load_default_collision_coupler()
        viewer.add_scaffold_model(model)
    viewer.show()

def stick_optimization(file_path):
    global viewer
    viewer = ScaffoldOptimizerViewer()
    if file_path is None or file_path == "":
        file_path = "one_tet.json"
    viewer.load_from_file(file_path)

    remote_tx = MqttTransport(REMOTE_SERVER_NAME)
    local_tx = MqttTransport(LOCAL_SERVER_NAME)

    topic = Topic("/scaffold/stick_model/", Message)
    subscriber_stick = Subscriber(topic, callback=update_viewer_stick_model, transport=remote_tx)
    subscriber_stick.subscribe()

    topic = Topic("/opt/scaffold_model/", Message)
    subscriber_scaffold = Subscriber(topic, callback=update_viewer_scaffold_models, transport=local_tx)
    subscriber_scaffold.subscribe()
    viewer.show()

def computation_process():

    local_tx = MqttTransport(LOCAL_SERVER_NAME)
    topic = Topic("/opt/problem_request/", Message)

    subscriber_stick = Subscriber(topic, callback=update_optimization, transport=local_tx)
    subscriber_stick.subscribe()

    print("Optimization waiting for messages, press CTRL+C to cancel")
    while True:
        time.sleep(1)



