import os

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'multi_tangent_data', 'frame_assembly'))
MT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'multi_tangent_data', 'mt_results'))
COUPLER_OBJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'coupler', 'coupler.obj'))
COUPLER_COARSE_OBJ_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'coupler', 'coupler_coarse.obj'))
COUPLER_COLLI_OBJ_0_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'coupler', 'half_coupler_0.obj'))
COUPLER_COLLI_OBJ_1_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'coupler', 'half_coupler_1.obj'))
COUPLER_COLLI_OBJ_2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'coupler', 'half_coupler_2.obj'))
COUPLER_COLLI_OBJ_PATHs = [COUPLER_COLLI_OBJ_0_PATH, COUPLER_COLLI_OBJ_1_PATH, COUPLER_COLLI_OBJ_2_PATH]