from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

if __name__ == "__main__":
    queue = Queue()

    p2 = Process(target=computation_process)
    p1 = Process(target=scaffold_visualization, args=(queue,))

    p1.start()

    file_path = None

    p2.start()
    queue.put(file_path)
    p1.join()

    if p2.is_alive():
        p2.join()