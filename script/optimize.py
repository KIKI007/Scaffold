import argparse
from scaffold.gui import ScaffoldOptimizerViewer, ScaffoldViewer
from scaffold.formfind.optimizer import SMILP_optimizer
from scaffold.io import StickModelInput, ScaffoldModelOutput
from multiprocessing import Process, Queue
import time

def scaffold_optimization_gui(file_path, compute_queue, draw_queue):
    viewer = ScaffoldOptimizerViewer()
    if file_path is None or file_path == "":
        file_path = "one_tet.json"
    viewer.load_from_file(file_path)
    viewer.draw_queue = draw_queue
    viewer.compute_queue = compute_queue
    viewer.show()

def computation_process(compute_queue, draw_queue):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Computational design and fabrication of reusable multi-tangent bar structures')
    parser.add_argument('--name', type=str, default = "box2x2", help='The name of input model (no extension)')

    args = parser.parse_args()
    file_path = args.name + ".json"

    draw_queue = Queue()
    compute_queue = Queue()
    p1 = Process(target=scaffold_optimization_gui, args=(file_path, compute_queue, draw_queue))
    p2 = Process(target=computation_process, args=(compute_queue, draw_queue))

    p1.start()
    p2.start()
    p1.join()
    p2.terminate()