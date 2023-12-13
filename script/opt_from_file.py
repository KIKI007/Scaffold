from scaffold.process_managers import *
import multiprocessing as mp
from multiprocessing import Process, Queue

if __name__ == "__main__":
    p2 = Process(target=computation_process)
    p1 = Process(target=gui_process)

    p2.start()
    p1.start()

    p1.join()
    p2.join()

