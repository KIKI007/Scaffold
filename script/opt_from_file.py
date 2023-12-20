from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-lp', #'--legacy_problem', default="one_tet.json",
                        help='legacy way of loading .json file')

    parser.add_argument("-p", '--problem', default="nonreciprocal_dome.json",
                        help='current way of loading .json file')

    args = parser.parse_args()

    queue = mp.Queue()

    p2 = Process(target=computation_process)
    p1 = Process(target=gui_process, args=(queue, ))

    p2.start()
    p1.start()

    data = {}

    if args.problem != None:
        data["file_type"] = "current"
        data["file"] = args.problem
    elif args.legacy_problem != None:
        data["file_type"] = "legacy"
        data["file"] = args.legacy_problem

    queue.put(data)

    p1.join()
    p2.join()

