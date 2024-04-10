from .geometry import ScaffoldModel
from pytransform3d.transform_manager import TransformManager
from distance3d.colliders import Capsule, MeshGraph
from distance3d.broad_phase import BoundingVolumeHierarchy
from distance3d.gjk import gjk
from distance3d.urdf_utils import fast_transform_manager_initialization
import numpy as np
import polyscope as ps

class CollisionSolver(ScaffoldModel):
    def __init__(self, model):
        super().__init__(model)

    def solve(self):
        collision_id = 0
        while True:
            if ps.has_surface_mesh("collision {}".format(collision_id)):
                ps.remove_surface_mesh("collision {}".format(collision_id))
                collision_id = collision_id + 1
            else:
                break

        ps.remove_group("collision", False)
        group = ps.create_group("collision")

        collision_margin = 1e-9
        tm = TransformManager(check=False)
        bvh = BoundingVolumeHierarchy(tm, "base")

        num_colliders = len(self.lines) + len(self.adj) * 6 * len(self.coupler_colliders)
        print(num_colliders)

        fast_transform_manager_initialization(tm, range(num_colliders), "base")

        collider_id = 0
        exclude_pairs = []
        objs = []

        for i in range(len(self.lines)):
            A2B, radius, height = self.beam_mat4(i)
            collider = Capsule(np.eye(4), radius, height)
            collider.make_artist()

            tm.add_transform(collider_id, "base", A2B)
            bvh.add_collider(collider_id, collider)
            collider_id = collider_id + 1

        if self.coupler_colliders != []:
            for coupler_id in range(len(self.adj)):
                coupler_collider_inds = []
                beam_inds = [*self.adj[coupler_id]]
                for side_id in range(2):
                    for mesh_id in range(len(self.coupler_colliders)):
                        A2B = self.coupler_mat4(coupler_id, side_id)
                        [mv, mf] = self.coupler_colliders[mesh_id]
                        collider = MeshGraph(np.eye(4), mv, mf)
                        collider.make_artist()

                        tm.add_transform(collider_id, "base", A2B)
                        bvh.add_collider(collider_id, collider)
                        coupler_collider_inds.append(collider_id)
                        collider_id = collider_id + 1

                for i in coupler_collider_inds:
                    for j in coupler_collider_inds:
                        exclude_pairs.append([i, j])

                for i in range(2):
                    for j in range(3):
                        exclude_pairs.append([beam_inds[i], coupler_collider_inds[3 * i + j]])
                        exclude_pairs.append([coupler_collider_inds[3 * i + j], beam_inds[i]])

        bvh.update_collider_poses()

        # add to visualization
        for artist in bvh.get_artists():
            for geometry in artist.geometries:
                v = np.asarray(geometry.vertices)
                f = np.asarray(geometry.triangles)
                objs.append([v, f])

        # check collision
        collisions_paris_id = 0
        pairs = bvh.aabb_overlapping_with_self()
        for (frame1, collider1), (frame2, collider2) in pairs:
            if [frame1, frame2] not in exclude_pairs and frame1 < frame2:
                dist, point1, point2, _ = gjk(collider1, collider2)
                if dist < collision_margin:

                    [v1, f1] = objs[frame1]
                    [v2, f2] = objs[frame2]

                    pair1 = ps.register_surface_mesh("collision pair {} 1".format(collisions_paris_id), v1, f1)
                    pair1.set_color((1, 1, 0))
                    pair2 = ps.register_surface_mesh("collision pair {} 2".format(collisions_paris_id), v2, f2)
                    pair2.set_color((1, 1, 0))

                    pair12 = ps.create_group("collision pair {}".format(collisions_paris_id))
                    pair1.add_to_group(pair12)
                    pair2.add_to_group(pair12)

                    pair12.set_enabled(False)
                    pair12.set_show_child_details(False)
                    pair12.set_hide_descendants_from_structure_lists(True)

                    collisions_paris_id = collisions_paris_id + 1

        objID = 0
        for [v, f] in objs:
            obj = ps.register_surface_mesh("zcollision {}".format(objID), v, f)
            obj.set_color((1, 0, 0))
            obj.set_transparency(0.1)
            obj.add_to_group(group)
            objID = objID + 1

        group.set_show_child_details(False)
        group.set_hide_descendants_from_structure_lists(True)