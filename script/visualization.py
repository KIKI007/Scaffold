from scaffold.process_managers import *
from multiprocessing import Process

if __name__ == "__main__":
    file_path = "bunny_layer_0.json"
    p1 = Process(target=scaffold_visualization, args=(file_path,))
    p1.start()

    p2 = Process(target=computation_process)
    p2.start()

    p1.join()
    p2.join()