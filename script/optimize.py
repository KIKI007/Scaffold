import argparse
from scaffold.compute import computation_process, scaffold_optimization_gui
from scaffold.gui import ScaffoldOptimizerViewer
from multiprocessing import Queue, Process

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