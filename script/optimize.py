from scaffold.process_managers import *
from multiprocessing import Process

if __name__ == "__main__":
    queue = Queue()

    file_path = ""
    p1 = Process(target=stick_optimization, args=(file_path, ))
    p1.start()

    p2 = Process(target=computation_process)
    p2.start()

    p1.join()
    p2.join()