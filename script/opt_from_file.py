from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-lp', '--legacy_problem', default="one_tet.json",
                        help='legacy way of loading stick model .json file')

    parser.add_argument("-p", '--problem',
                        help='current way of loading stick model .json file')

    parser.add_argument("-v", '--visualization',
                        help='current way of loading scaffold model .json file')

    args = parser.parse_args()

    queue = mp.Queue()

    p2 = Process(target=computation_process)
    p1 = Process(target=gui_process, args=(queue, ))

    p1.start()

    data = {}

    if args.visualization != None:
        data["file_type"] = "visualization"
        data["file"] = args.visualization
    elif args.problem != None:
        data["file_type"] = "current"
        data["file"] = args.problem
        p2.start()
    elif args.legacy_problem != None:
        data["file_type"] = "legacy"
        data["file"] = args.legacy_problem
        p2.start()

    queue.put(data)

    p1.join()
    if p2.is_alive():
        p2.join()