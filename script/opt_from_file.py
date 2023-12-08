from scaffold.io import StickModelImporter, ScaffoldModelImporter
from scaffold.gui import ScaffoldViewer, ScaffoldOptimizerViewer
from scaffold.collision import CollisionChecker
import time
from scaffold.formfind.optimizer import SMILP_optimizer

if __name__ == "__main__":

    viewer = ScaffoldOptimizerViewer("twoboxes")
    viewer.load_from_file()

    viewer.register_model(viewer.stick_model)

    viewer.show()
