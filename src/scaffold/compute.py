from scaffold.formfind.optimizer import SMILP_optimizer
from scaffold.io import StickModelInput
from scaffold.gui import ScaffoldOptimizerViewer
from multiprocessing import Queue
import time

def computation_process(compute_queue: Queue, draw_queue : Queue):
    while True:
        try:
            request_json = compute_queue.get(block=False)
            input = StickModelInput()
            input.fromJSON(request_json)
            optimizer = SMILP_optimizer(input.stick_model.file_name)
            optimizer.input_model(input)
            optimizer.draw_queue = draw_queue
            optimizer.solve()
        except:
            pass
        time.sleep(0.1)

def scaffold_optimization_gui(file_path, compute_queue, draw_queue):
    viewer = ScaffoldOptimizerViewer()
    if file_path is None or file_path == "":
        file_path = "one_tet.json"
    viewer.load_from_file(file_path)
    viewer.draw_queue = draw_queue
    viewer.compute_queue = compute_queue
    viewer.show()

def scaffold_optimization_gui_rhino(stick_model,
                                    compute_queue: Queue,
                                    draw_queue: Queue,
                                    rhino_queue:Queue):

    viewer = ScaffoldOptimizerViewer()
    viewer.input = stick_model
    viewer.re_render = True
    viewer.rhino = True
    viewer.draw_queue = draw_queue
    viewer.compute_queue = compute_queue
    viewer.rhino_queue = rhino_queue
    viewer.send_optimization_command()
    viewer.show()