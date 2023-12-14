from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='geodesic_dome.json',
                        help='The name of the frame json problem to solve')
    args = parser.parse_args()

    queue = mp.Queue()

    p2 = Process(target=computation_process)
    p1 = Process(target=gui_process, args=(queue, ))

    p2.start()
    p1.start()
    queue.put(args.problem)

    p1.join()
    p2.join()

