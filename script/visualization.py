from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

if __name__ == "__main__":
    queue = Queue()

    file_path = "box3x3_layer_0.json"
    p1 = Process(target=scaffold_visualization, args=(file_path,))
    p1.start()

    p2 = Process(target=computation_process)
    p2.start()

    p1.join()
    p2.join()