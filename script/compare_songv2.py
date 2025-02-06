import argparse, os
from collections import defaultdict
import json

import numpy as np
from numpy.linalg import norm as npnorm
import jax.numpy as jnp
from jax import grad, jit
from functools import partial
import scipy

import matplotlib.pyplot as plt
from matplotlib import collections as plt_collections

import pybullet_planning as pp

def compute_cyclic_contact_assignment(edges):
    beam_ids_from_vertex = defaultdict(set)
    for e_id, e in enumerate(edges):
        beam_ids_from_vertex[e[0]].add(e_id)
        beam_ids_from_vertex[e[1]].add(e_id)
    contact_assignments = []
    for v_id, v_edge_set in beam_ids_from_vertex.items():
        v_edges = list(v_edge_set)
        if len(v_edges) > 1:
            contact_ring = []
            if len(v_edges) == 2:
                contact_ring.append(tuple(v_edges))
                # contact_assignments.expend(tuple(v_edges))
            else:
                for kd in range(len(v_edges)):
                    e0 = v_edges[kd]
                    e1 = v_edges[(kd+1) % len(v_edges)]
                    contact_ring.append((e0, e1))
                    # contact_assignments.expend((e0, e1))
            contact_assignments.append(contact_ring)
    return contact_assignments

def normalize(v):
    vnorm = npnorm(v)
    if vnorm < np.finfo(v.dtype).eps: 
        print('normalize vector with norm close to zero!')
        raise
        # return v
        vnorm = np.finfo(v.dtype).eps
    return v / vnorm

def get_edge_direction(edge, vertices):
    v0, v1 = edge
    p0 = np.array(vertices[v0,:])
    p1 = np.array(vertices[v1,:])
    return normalize(p1 - p0)

def divide_contact_beam_pairs(vertices, edges, contact_assignments, sinEps=1e-3):
    """_summary_

    Parameters
    ----------
    vertices: num_nodes x 3 np array
    edges:    num_edges x 2 np array
    sinEps : float, optional
        eps value for determining line parellel, by default 1e-3

    Returns
    -------
    mat_non_parallel_e_pairs
        2 x pair_size int matrix, non parallel connected line pair ids
    non_parallel_e_signs
        pair_size int list, non parallel connected line pair's sign, can take value 1 () or -1 ()
    mat_parallel_e_pairs
        2 x pair_size int matrix: parallel connected line pair ids
    mat_parallel_e_perp_dirs
        3 x pair_size int matrix: parallel connected line pair's perpedicular directions
    """
    non_parallel_e_pairs = []
    non_parallel_e_signs = []
    parallel_e_pairs = [] 
    parallel_e_perp_dirs = []

    for contact_ring in contact_assignments:
        dir_cross = []
        for i, (e0, e1) in enumerate(contact_ring):
            d0 = get_edge_direction(edges[e0,:], vertices)
            d1 = get_edge_direction(edges[e1,:], vertices)
            dir_cross.append(np.cross(d0,d1))
            # if two lines are in parellel
            if npnorm(np.cross(d0, d1)) < sinEps:
                drt = np.array([0,0,1.0])
                # find a direction that's not parallel to d0
                if npnorm(np.cross(drt,d0)) < sinEps:
                    drt = np.array([1.0,0,0])
                    if npnorm(np.cross(drt,d0)) < sinEps:
                        raise ValueError('z and x axes both parallel to e{}'.format(e0))
                parallel_e_pairs.append([e0,e1])
                parallel_e_perp_dirs.append(drt)
            else: # non-parellel
                non_parallel_e_pairs.append((e0,e1))
                if len(contact_ring) >= 3 and i == len(contact_ring)-1 and np.dot(dir_cross[-1], dir_cross[-2]) >= 0:
                    non_parallel_e_signs.append(-1)
                else:
                    non_parallel_e_signs.append(1)

    mat_non_parallel_e_pairs = np.empty((2, len(non_parallel_e_pairs)), dtype=int)
    for i, ep in enumerate(non_parallel_e_pairs):
        mat_non_parallel_e_pairs[:,i] = ep

    mat_parallel_e_pairs = np.empty((2, len(parallel_e_pairs)), dtype=int)
    for i, ep in enumerate(parallel_e_pairs):
        mat_parallel_e_pairs[:,i] = ep

    mat_parallel_e_perp_dirs = np.empty((3, len(parallel_e_perp_dirs)), dtype=float)
    for i, ed in enumerate(parallel_e_perp_dirs):
        mat_parallel_e_perp_dirs[:,i] = ed

    return mat_non_parallel_e_pairs, non_parallel_e_signs, \
        mat_parallel_e_pairs, mat_parallel_e_perp_dirs

########################

class BeamOpt:
    def __init__(self, vertices, edges, support_vertex_indices, radius, contact_assigments=None):
        """
        Inputs
        ======
        vertices: num_nodes x 3 np array
        edges:    num_edges x 2 np array
        support_vertex_indices: num_supp x 1 np array
        """
        # vertices' coordinate
        self.vertices = jnp.array(vertices.copy())

        # beams' vertex ids
        self.edges = jnp.array(edges.copy())

        # fixed vertex ids
        self.support_vertex_indices = support_vertex_indices.copy()

        self.end_pts_init_positions = (self.vertices[self.edges]).flatten().copy()

        # TODO allow certain edges to be frozen

        self.c_assigns = contact_assigments or compute_cyclic_contact_assignment(edges)
        [_non_parallel_e_pairs, _non_parallel_e_signs, _parallel_e_pairs, _parallel_e_perp_dirs] = \
            divide_contact_beam_pairs(vertices, edges, self.c_assigns, 1e-3)

        # from multi_tangent.archive import computeOffsetBeamConstraintsPairs
        # [_non_parallel_e_pairs, _non_parallel_e_signs, _parallel_e_pairs, _parallel_e_perp_dirs] = \
        #     computeOffsetBeamConstraintsPairs(vertices, edges, 0.2)
        #     # assembly.computeOffsetBeamConstraintsPairs(0.2) # C++ version

        #beam beam connection
        # 2xpair non-parallel pairs
        self.non_parallel_e_pairs = jnp.array(_non_parallel_e_pairs)
        # non-parallel pair sign
        self.non_parallel_e_signs = jnp.array(_non_parallel_e_signs)

        # 2xpair parallel pairs
        self.parallel_e_pairs = jnp.array(_parallel_e_pairs)
        # parallel pairs' perpendicular directions
        self.parallel_e_perp_dirs = jnp.array(_parallel_e_perp_dirs)

        self.nB = self.edges.shape[0]
        self.nE_np = self.non_parallel_e_pairs.shape[1]
        self.nE_p = self.parallel_e_pairs.shape[1]

        # nT: number of variables for beam's intersection point
        self.nT = (self.nE_np + self.nE_p) * 2

        # nP: number of varibles Pi for beams' endpoints
        self.nP = self.end_pts_init_positions.shape[0]

        self.fix_var_ids = self.nT + np.array([[sv*3, sv*3+1, sv*3+2] for sv in self.support_vertex_indices], dtype=int).flatten()

        # TODO: multiple materials and radius
        self.radius = radius
        self.weights = [1, 30, 1]

        self.xstar = self.computeInitVar()
        self.x_trace = []

    def computeInitVar(self):
        t = jnp.ones(self.nT) * 0.5
        var = jnp.hstack([t, self.end_pts_init_positions]).copy()
        # reset trace
        self.x_trace = []
        return var

    def checkBeamDistanceConstraints(self, var):
        return computeBeamDistanceEnergy(var, self)

    def callback(self, xk):
        self.x_trace.append(xk)

    def runOptimization(self, trace=True):
        """Run unconstrained optimization with Newton-CG for the given contact pattern

        Returns
        -------
        new_V: (num_beam x 2) x 3 np array for beam vertices, each row is [x,y,z]
        new_EV: (num_beam) x 2 np array for the edge assignment, 
            the row ordering corresponds with the original edges, but the vertex indices mapping is reordered.
        new_fixed_vert_ids: list of int
            fixed vertex ids in new_V
        """
        var = self.computeInitVar()
        result = scipy.optimize.minimize(computeBeamOptObjective,
                                         var,
                                         args=self,
                                         jac=computeBeamOptObjectiveGradient,
                                         hessp=computeBeamOptObjectiveHessianP,
                                         method="Newton-CG",
                                         callback=self.callback,
                                         options={
                                            'disp':True,
                                            'return_all':trace,
                                                  })

        print(f'Opt success: {result.success} | Status: {result.status} | {result.message} | n_iter: {result.nit}')
        # print(f'nfev, njev, nhev : {result.nfev}, {result.njev}, {result.nhev}')
        print('Distance constraint: {}'.format(self.checkBeamDistanceConstraints(result.x)))
        self.xstar = result.x
        new_V = result.x[self.nT: self.nP + self.nT].reshape(self.nB * 2, 3)
        new_EV = np.array([[2 * i, 2 * i + 1] for i in range(0, self.nB)], dtype=int)
        # if trace:
        #     self.x_trace = result.allvecs

        # * map the old fixed node ids into the new ones
        old_fixed_beam_end_id = defaultdict(list)
        for gv in self.support_vertex_indices:
            for e_id, e in enumerate(self.edges):
                for i, end_id in enumerate(e):
                    if gv == end_id:
                        old_fixed_beam_end_id[gv].append((e_id, i))

        new_fixed_vert_ids = set()
        for _, e_endpts in old_fixed_beam_end_id.items():
            for e, i in e_endpts:
                new_fixed_vert_ids.add(new_EV[e][i])

        # new_assembly = BeamAssembly(new_V, new_EV, self.new_fixed_vert_ids)
        # new_assembly.setMaterial(self.material)
        # new_assembly.update()
        return new_V, new_EV, list(new_fixed_vert_ids)

    @property
    def new_fixed_vert_ids(self):
        """map the old fixed node ids into the new ones
        """
        new_EV = [[2 * x, 2 * x + 1] for x in range(0, self.nB)]
        old_fixed_beam_end_id = defaultdict(list)
        for gv in self.support_vertex_indices:
            for e_id, e in enumerate(self.edges):
                for i, end_id in enumerate(e):
                    if gv == end_id:
                        old_fixed_beam_end_id[gv].append((e_id, i))
        new_fixed_vert_ids = set()
        for _, e_endpts in old_fixed_beam_end_id.items():
            for e, i in e_endpts:
                new_fixed_vert_ids.add(new_EV[e][i])
        return list(new_fixed_vert_ids)


################################################

@partial(jit, static_argnums=(1,))
def computeBeamDistanceEnergy(var, beam_opt: BeamOpt):
    # non-parallel intersection point vars (line parameterization t)
    tE = var[0: beam_opt.nE_np * 2].reshape(2, beam_opt.nE_np)
    # parallel intersection point vars
    tE2 = var[beam_opt.nE_np * 2 : beam_opt.nE_np * 2 + beam_opt.nE_p * 2].reshape(2, beam_opt.nE_p)
    # end point per beam vars
    P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    # nB x 3
    drt = P[:, 1, :] - P[:, 0, :]

    origin = P[:, 0, :]
    energy = 0
    # non-parallel pairs
    normal = jnp.cross(drt[beam_opt.non_parallel_e_pairs[0, :], :], drt[beam_opt.non_parallel_e_pairs[1, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1) * beam_opt.non_parallel_e_signs
    normal = jnp.diag(inv_length) @ normal
    # (nE_np x nE_np) * (nE_np, 3)
    # nB x 3
    p0 = jnp.diag(tE[0, :]) @ drt[beam_opt.non_parallel_e_pairs[0, :], :] + origin[beam_opt.non_parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE[1, :]) @ drt[beam_opt.non_parallel_e_pairs[1, :], :] + origin[beam_opt.non_parallel_e_pairs[1, :], :]
    energy = jnp.vdot(p1 - p0 - normal * beam_opt.radius * 2, p1 - p0 - normal * beam_opt.radius * 2)
    # TODO lower bound the distance to prevent penetration

    # parallel pairs
    normal = jnp.cross(beam_opt.parallel_e_perp_dirs.transpose(), drt[beam_opt.parallel_e_pairs[0, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1)
    normal = jnp.diag(inv_length) @ normal
    p0 = jnp.diag(tE2[0, :]) @ drt[beam_opt.parallel_e_pairs[0, :], :] + origin[beam_opt.parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE2[1, :]) @ drt[beam_opt.parallel_e_pairs[1, :], :] + origin[beam_opt.parallel_e_pairs[1, :], :]
    energy += jnp.vdot(p1 - p0 - normal * beam_opt.radius * 2, p1 - p0 - normal * beam_opt.radius * 2)
    # ? for ensuring parallel
    energy += jnp.vdot(normal, drt[beam_opt.parallel_e_pairs[1, :], :]) ** 2

    return energy

@partial(jit, static_argnums=(1,))
def computeBeamOptObjective(var, beam_opt: BeamOpt):
    t = var[0: beam_opt.nT].reshape(2, (beam_opt.nE_np + beam_opt.nE_p))
    Pf = var[beam_opt.nT: beam_opt.nT + beam_opt.nP]
    # P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    # * origianl graph approximation energy
    energy1 = jnp.vdot(beam_opt.end_pts_init_positions - Pf, beam_opt.end_pts_init_positions - Pf)
    # energy1 = 0.0
    energy2 = computeBeamDistanceEnergy(var, beam_opt)
    # * parameter t range
    energy3 = jnp.vdot(jnp.clip(t, 0.1, 0.9) - t, jnp.clip(t, 0.1, 0.9) - t)

    # * Fix supports
    # energy4 = jnp.vdot(beam_opt.end_pts_init_positions[beam_opt.fix_var_ids] - Pf[beam_opt.fix_var_ids], 
    #                    beam_opt.end_pts_init_positions[beam_opt.fix_var_ids] - Pf[beam_opt.fix_var_ids])

    summed_energy = beam_opt.weights[0] * energy1 + beam_opt.weights[1] * energy2 + beam_opt.weights[2] * energy3
    # + energy4*10
    return summed_energy

@partial(jit, static_argnums=(1,))
def computeBeamOptObjectiveGradient(var, beamOpt: BeamOpt):
    return grad(computeBeamOptObjective, 0)(var, beamOpt)

@partial(jit, static_argnums=(2,))
def computeGradientV(var, v, beamOpt: BeamOpt):
    gradient = computeBeamOptObjectiveGradient(var, beamOpt)
    return jnp.dot(gradient, v)

@partial(jit, static_argnums=(2,))
def computeBeamOptObjectiveHessianP(var, v, beamOpt: BeamOpt):
    return grad(computeGradientV, 0)(var, v, beamOpt)

##############################################

def compute_distance_energy_per_connector(var, beam_opt: BeamOpt):
    # non-parallel intersection point vars (line parameterization t)
    tE = var[0: beam_opt.nE_np * 2].reshape(2, beam_opt.nE_np)
    # parallel intersection point vars
    tE2 = var[beam_opt.nE_np * 2 : beam_opt.nE_np * 2 + beam_opt.nE_p * 2].reshape(2, beam_opt.nE_p)
    # end point per beam vars
    P = var[beam_opt.nT: beam_opt.nT + beam_opt.nP].reshape(beam_opt.nB, 2, 3)
    drt = P[:, 1, :] - P[:, 0, :]

    origin = P[:, 0, :]
    energy = 0
    # non-parallel pairs
    normal = jnp.cross(drt[beam_opt.non_parallel_e_pairs[0, :], :], drt[beam_opt.non_parallel_e_pairs[1, :], :])
    # normalization + sign
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1) * beam_opt.non_parallel_e_signs
    normal = jnp.diag(inv_length) @ normal

    p0 = jnp.diag(tE[0, :]) @ drt[beam_opt.non_parallel_e_pairs[0, :], :] + origin[beam_opt.non_parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE[1, :]) @ drt[beam_opt.non_parallel_e_pairs[1, :], :] + origin[beam_opt.non_parallel_e_pairs[1, :], :]
    # energy = jnp.vdot(p1 - p0 - normal * beam_opt.width, p1 - p0 - normal * beam_opt.width)
    np_vec_diff = p1 - p0 - normal * beam_opt.radius * 2

    # parallel pairs
    normal = jnp.cross(beam_opt.parallel_e_perp_dirs.transpose(), drt[beam_opt.parallel_e_pairs[0, :], :])
    inv_length = 1.0 / jnp.linalg.norm(normal, axis = 1)
    normal = jnp.diag(inv_length) @ normal
    p0 = jnp.diag(tE2[0, :]) @ drt[beam_opt.parallel_e_pairs[0, :], :] + origin[beam_opt.parallel_e_pairs[0, :], :]
    p1 = jnp.diag(tE2[1, :]) @ drt[beam_opt.parallel_e_pairs[1, :], :] + origin[beam_opt.parallel_e_pairs[1, :], :]
    # energy += jnp.vdot(p1 - p0 - normal * beam_opt.width, p1 - p0 - normal * beam_opt.width)
    p_vec_diff = p1 - p0 - normal * beam_opt.radius * 2
    # energy += jnp.vdot(normal, drt[beam_opt.parallel_e_pairs[1, :], :]) ** 2

    return np.vstack([np_vec_diff, p_vec_diff])

##############################################

def convex_combination(x1, x2, w=0.5):
    assert 0 <= w and w <= 1
    return (1 - w) * np.array(x1) + (w * np.array(x2))

def closest_point_segments(line_point_1_1, line_point_1_2, line_point_2_1, line_point_2_2, opt_tol=1e-8, use_knitro=False):
    t1, t2 = compute_closest_t_between_lines(line_point_1_1, line_point_1_2, line_point_2_1, line_point_2_2, opt_tol=opt_tol)
    return [convex_combination(line_point_1_1, line_point_1_2, t1),\
            convex_combination(line_point_2_1, line_point_2_2, t2)]

# distance between a point and a line segment
def compute_closest_point_to_line(point, line_point_1, line_point_2, opt_tol=1e-8):
    p1 = line_point_1
    p2 = line_point_2
    d = p2 - p1
    assert np.linalg.norm(d) > opt_tol, 'degenerate line segment'

    t = -(p1 - point).dot(d) / d.dot(d)
    return np.clip(t, 0, 1)

def compute_closest_t_between_lines(line_point_1_1, line_point_1_2, line_point_2_1, line_point_2_2, opt_tol=1e-8):
    p1 = line_point_1_1
    p2 = line_point_1_2
    p3 = line_point_2_1
    p4 = line_point_2_2
    d1 = p2 - p1
    d2 = p4 - p3
    assert np.linalg.norm(d1) > opt_tol and np.linalg.norm(d2) > opt_tol, 'degenerate line segment'

    A = np.array([[d1.dot(d1), -d1.dot(d2)], 
                  [d2.dot(d1), -d2.dot(d2)]])
    denominator = np.linalg.det(A)
    b = np.array([(p3 - p1).dot(d1), (p3 - p1).dot(d2)])

    # if two lines are not parallel
    if abs(denominator) > opt_tol:
        t = np.linalg.inv(A) @ b
        # intersection happens inbetween the two line segments
        if t[0] >= 0 and t[0] <= 1 and t[1] >=0 and t[1] <= 1:
            return t

    #the closest points must include a end point of the two line segments
    ts = [[0, None], [1, None], [None, 0], [None, 1]]
    min_dist = np.inf
    min_t = []
    for tc in ts:
        t = []
        if tc[0] == None:
            pt = convex_combination(p3, p4, tc[1])
            t0 = compute_closest_point_to_line(pt, p1, p2, opt_tol)
            t = [t0, tc[1]]
        elif tc[1] == None:
            pt = convex_combination(p1, p2, tc[0])
            t1 = compute_closest_point_to_line(pt, p3, p4, opt_tol)
            t = [tc[0], t1]

        dist = np.linalg.norm(convex_combination(p1, p2, t[0]) - convex_combination(p3, p4, t[1]))
        #print(t, dist)
        if min_dist > dist:
            min_t = t
            min_dist = dist

    return min_t

def plot_contact_distance(vertices, edges, c_assigns, radius):
    fig, axes = plt.subplots(1,1, figsize=(12,6))
    # clean up duplicated bidirectional edges
    dist_data = []
    for c_rings in c_assigns:
        for c in c_rings:
            e1, e2 = c
            # if n1 == GROUND_INDEX or n2 == GROUND_INDEX:
            #     continue
            # reversed_edge = edge[::-1]
            # if reversed_edge in dist_data:
            #     continue
            # dist_data[edge] = (npnorm(np.array(c[0]) - np.array(c[1])))
            # radius_lines[edge] = ([(count,2*radius), (count+1,2*radius)])
            contact_pts = closest_point_segments(vertices[edges[e1,0],:], vertices[edges[e1,1],:], vertices[edges[e2,0],:], vertices[edges[e2,1],:])
            dist_data.append(npnorm(contact_pts[0] - contact_pts[1]))

    # * histogram
    # axes[0].hist(dist_data, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    # axes[0].set_xlabel('Shortest distance between contact bars [m]')
    # axes[0].set_ylabel('Count')
    # axes[0].ticklabel_format(useOffset=False, style='plain')

    # * line plot
    # lc = plt_collections.LineCollection(list(radius_lines.values()), colors='blue', linewidths=2)
    # axes[1].add_collection(lc)
    axes.plot(dist_data)
    axes.plot([(2*radius) for _ in range(len(dist_data))], color='green', linewidth=2)
    axes.set_xlabel('Connector index')
    axes.set_ylabel('Dist between contact bars [m]')
    axes.ticklabel_format(useOffset=False, style='plain')
    axes.set_title('Contact dist from BarStructure')

    return fig

def plot_beam_opt_per_pair(beam_opt: BeamOpt):
    x_vecdiff = compute_distance_energy_per_connector(beam_opt.xstar, beam_opt)
    x_energy = npnorm(x_vecdiff, axis=1)

    fig, axes = plt.subplots(1,2, figsize=(12,6))

    # * histogram
    axes[0].hist(x_energy, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    axes[0].set_xlabel('$(E_1(t_1) - E_2(t_2)) \cdot 2r*n$')
    axes[0].set_ylabel('Count')

    # * line plot
    axes[1].plot([(beam_opt.radius*2)**2 for _ in range(len(x_energy))], color='blue', linewidth=2)
    axes[1].plot(x_energy, 'bo')
    axes[1].set_xlabel('Connector index')
    axes[1].set_ylabel('vec product')
    fig.suptitle('Contact dist from beam_opt $x_*$')

    return fig

def plot_f_trace(beam_opt: BeamOpt):
    f_sum_trace = [computeBeamOptObjective(x, beam_opt) for x in beam_opt.x_trace]
    f_beam_trace = [computeBeamDistanceEnergy(x, beam_opt) for x in beam_opt.x_trace]

    fig, axes = plt.subplots(figsize=(12,6))
    # * histogram
    # axes[0].hist(x_energy, 20, lw=2, ec="yellow", fc="green", alpha=0.5)
    # axes[0].set_xlabel('$(E_1(t_1) - E_2(t_2)) \cdot 2r*n$')
    # axes[0].set_ylabel('Count')
    # * line plot
    axes.plot(f_sum_trace, label='summed energy', color='blue', linewidth=2)
    axes.plot(f_beam_trace, label='beam distance energy', color='red', linewidth=2)
    axes.set_xlabel('iter')
    axes.set_ylabel('energy')
    axes.legend()
    print(list(f_sum_trace))
    print(list(f_beam_trace))
    # fig.suptitle('Contact dist from beam_opt $x_*$')
    return fig

def create_bar_body(_p1, _p2, bar_radius, scale=1.0, use_box=False, color=pp.apply_alpha(pp.RED, 1), shrink_radius=0.0):
    """create bar's collision body in pybullet
    """
    if not pp.is_connected():
        return None
    p1 = np.array(_p1) * scale
    p2 = np.array(_p2) * scale
    # height = max(np.linalg.norm(p2 - p1) - 2*shrink, 0)
    height = max(np.linalg.norm(p2 - p1), 0)
    center = (p1 + p2) / 2

    delta = p2 - p1
    x, y, z = delta
    phi = np.math.atan2(y, x)
    theta = np.math.acos(z / np.linalg.norm(delta))
    quat = pp.quat_from_euler(pp.Euler(pitch=theta, yaw=phi))
    # p1 is z=-height/2, p2 is z=+height/2
    diameter = 2*(bar_radius*scale - shrink_radius)

    if use_box:
        # Much smaller than cylinder
        # use inscribed square
        h = diameter / np.sqrt(2)
        body = pp.create_box(h, h, height, color=color, mass=pp.STATIC_MASS)
    else:
        # Visually, smallest diameter is 2e-3
        # The geometries and bounding boxes seem correct though
        body = pp.create_cylinder(diameter/2, height, color=color, mass=pp.STATIC_MASS)
        # print('Diameter={:.5f} | Height={:.5f}'.format(diameter/2., height))
        # print(get_aabb_extent(get_aabb(body)).round(6).tolist())
        # print(get_visual_data(body))
        # print(get_collision_data(body))

    pp.set_point(body, center)
    pp.set_quat(body, quat)
    pp.set_color(body, color)
    # draw_aabb(get_aabb(body))
    # draw_pose(get_pose(body), length=5e-3)
    return body

##############################################

def main():
    np.set_printoptions(precision=3)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--problem', default='truss_one_tet_skeleton.json',
    #                     help='The name of the problem to solve')
    # parser.add_argument('--file_format', default='bar_structure', # assembly
    #                     help='File formats.')
    # parser.add_argument('--radius', default=None, type=float,
    #                     help='Bar radius')
    # parser.add_argument('-v', '--viewer', action='store_true', help='Enables the viewer during planning (slow!)')
    # parser.add_argument('-w', '--write', action='store_true', help='Export file.')
    # # parser.add_argument('-c', '--collisions', action='store_false',
    # #                     help='Disable collision checking with obstacles')
    # parser.add_argument('-db', '--debug', action='store_true', help='Debug verbose mode')
    # args = parser.parse_args()

    import json

    # Path to the JSON file
    file_name = 'box2x2'
    json_file_path = r"E:\Code\husky_ws\Scaffold\data\input\{}.json".format(file_name)
    with open(json_file_path, "r") as file:
        data = json.load(file)
    vertices = np.array(data["stick_model"]["vertices"])
    edges = np.array(data["stick_model"]["edges"], dtype=int)
    fixed_elements = data["mt_config"]["fixed_element_ids"]

    # read contact assignment from E:\Code\husky_ws\Scaffold\data\mt\roboarch_layer_0.json
    # json_file_path = r"E:\Code\husky_ws\Scaffold\data\mt\{}_layer_0.json".format(file_name)
    # with open(json_file_path, "r") as file:
    #     data = json.load(file)
    # contacts = []
    # for coupler in data['half_couplers']:
    #     c = coupler['at_element'], coupler['to_element']
    #     if c in contacts or c[::-1] in contacts:
    #         continue
    #     contacts.append([c])

    json_file_path = r"E:\Code\crl_ws\FrameX\data\mt_results\box_array_3d_2_MT_layer_0.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    contacts = []
    for c in data['contact_id_pairs']:
        if c in contacts or c[::-1] in contacts:
            continue
        contacts.append([c])

    # json_file_path = r"E:\Code\crl_ws\FrameX\data\stick_models\roboarch_florian.json"
    # json_file_path = r"E:\Code\crl_ws\FrameX\data\stick_models\box2x2.json"
    # with open(json_file_path, "r") as file:
    #     data = json.load(file)
    # vertices = np.array([n['point'] for n in data["nodes"]])
    # edges = np.array([e['end_node_inds'] for e in data["elements"]], dtype=int)
    # fixed_elements = data["fixed_element_ids"]

    support_vertex_indices = []
    for e in fixed_elements:
        support_vertex_indices.extend(edges[e])
    radius = 0.01

    print('Vertices: ', vertices.shape)
    print('Edges: ', edges.shape)

    beam_opt = BeamOpt(vertices, edges, support_vertex_indices, radius, contact_assigments=contacts)
    beam_opt.weights =[1, 200, 1]
    var = beam_opt.computeInitVar()
    fun = computeBeamOptObjective(var, beam_opt)
    gradient = computeBeamOptObjectiveGradient(var, beam_opt)
    v = jnp.ones(beam_opt.nP + beam_opt.nT)
    hessian = computeBeamOptObjectiveHessianP(var, v, beam_opt)
    new_vertices, new_edges, new_supports = beam_opt.runOptimization()

    # fig = plot_beam_opt_per_pair(beam_opt)
    # plt.show()
    fig = plot_f_trace(beam_opt)
    plt.show()
    fig = plot_contact_distance(new_vertices, new_edges, beam_opt.c_assigns, radius)
    plt.show()

    pp.connect(use_gui=True)
    pp.draw_pose(pp.unit_pose(), length=1.0)
    # # set_camera([[0,0,0]], camera_dist=0.5, scale=1.0)

    for e in new_edges:
        create_bar_body(new_vertices[e[0]], new_vertices[e[1]], radius, scale=1.0, color=pp.apply_alpha(pp.BLUE, 0.5))

    # if args.write:
    #     export_file_name = args.problem
    #     if 'skeleton' in export_file_name:
    #         export_file_name = export_file_name.split('_skeleton')[0] + '.json'
    #     if 'raw_bar_graph' in export_file_name:
    #         export_file_name = export_file_name.split('_raw_bar_graph')[0] + '.json'
    #     fig.savefig(os.path.join(multi_tangent.DATA_DIRECTORY, 'plots', export_file_name + '_bar_structure_dist_plot.png'))
    # plt.show()

    # * plot reference graph
    with pp.LockRenderer():
        # pp.draw_pose(pp.unit_pose(), length=0.01)
        # label_points([pt*SCALE for pt in node_points])
        for e in edges:
            n1, n2 = e
            p1 = vertices[n1]
            p2 = vertices[n2]
            pp.add_line(p1, p2, color=pp.apply_alpha(pp.BLUE, 0.3), width=0.5)
        for v in support_vertex_indices:
            pp.draw_circle(vertices[v], 0.01)

    # # check_model(bar_struct, debug=0, to_meter_scale=1.0, contact_dist_tol=3e-3, collision_dist_tol=0.002)
    contact_pts = []
    for c_rings in beam_opt.c_assigns:
        for c in c_rings:
            e1, e2 = c
            cpts = closest_point_segments(new_vertices[new_edges[e1,0],:], new_vertices[new_edges[e1,1],:], new_vertices[new_edges[e2,0],:], new_vertices[new_edges[e2,1],:])
            contact_pts.append([cpts[0].tolist(), cpts[1].tolist()])

    data = {
        "line": [[new_vertices[e[0]].tolist(), new_vertices[e[1]].tolist()] for e in new_edges],
        "radius" : radius,
        "contact_pts": contact_pts,
    }
    # save to ../data/mt/

    with open(r'E:\Code\husky_ws\Scaffold\data\mt\{}_peng.json'.format(file_name), 'w') as outfile:
        json.dump(data, outfile)

    pp.wait_if_gui()
    # pp.reset_simulation()
    # pp.disconnect()

if __name__ == '__main__':
    main()