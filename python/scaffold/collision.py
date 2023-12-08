import mujoco

class CollisionChecker:

    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def check(self):
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        if self.data.ncon != 0:
            print("Collision")