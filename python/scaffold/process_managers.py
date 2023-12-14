import json
from compas_eve import Message
from compas_eve import Subscriber
from compas_eve import Publisher
from compas_eve import Topic
from compas_eve.mqtt import MqttTransport
from scaffold.gui import ScaffoldOptimizerViewer
from scaffold.formfind.optimizer import SMILP_optimizer
from scaffold.geometry import StickModel, ScaffoldModel
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

def update_viewer_stick_model(msg):
    global viewer
    if viewer.running == False:
        model = StickModel()
        model.fromJSON(msg)

        viewer.stick_model = model
        viewer.opt_parameters["layers"] = msg["layers_ind"]
        viewer.opt_parameters["fixed_element_ids"] = msg["fixed_element_inds"]
        viewer.total_changed = True
    #viewer.register_model(model)

def update_viewer_scaffold_models(msg):
    global viewer
    if viewer.running == True:
        viewer.running_msg = msg["msg"]
        if msg["msg"] == "fail" or msg["msg"] == "succeed":
            viewer.running = False
            return

        model = ScaffoldModel()
        model.fromJSON(msg)
        viewer.refresh = True
        viewer.models.append(model)

def update_optimization(msg):

    optimizer = SMILP_optimizer(msg["name"])
    print(msg)
    optimizer.parse_from_json(msg)
    optimizer.solve()

def gui_process(queue):
    global viewer
    filename = queue.get()

    viewer = ScaffoldOptimizerViewer()
    viewer.load_from_file(filename)
    tx = MqttTransport("localhost")

    topic = Topic("/scaffold/stick_model/", Message)
    subscriber_stick = Subscriber(topic, callback=update_viewer_stick_model, transport=tx)
    subscriber_stick.subscribe()

    topic = Topic("/opt/scaffold_model/", Message)
    subscriber_scaffold = Subscriber(topic, callback=update_viewer_scaffold_models, transport=tx)
    subscriber_scaffold.subscribe()

    viewer.show()

def computation_process():

    tx = MqttTransport("localhost")
    topic = Topic("/opt/problem_request/", Message)
    subscriber_stick = Subscriber(topic, callback=update_optimization, transport=tx)
    subscriber_stick.subscribe()

    print("Optimization waiting for messages, press CTRL+C to cancel")
    while True:
        time.sleep(1)



