import math
from jax import grad, jit, vmap
import numpy as np
import jax.numpy as jnp
import os
from scaffold import MT_DIR
from termcolor import cprint
import json

def list_to_pairs(inds):
    assert len(inds) % 2 == 0
    return [(inds[i], inds[i+1]) for i in range(0,len(inds),2)]

def listify(d):
    return list(map(list, d))

def write_json_smilp(opt_data,
                     line_pts,
                     coupler_endpts,
                     smilp_parameters,
                     problem_name,
                     contact_opt = False,
                     layer_id = None):

    if problem_name.endswith('.json'):
        problem_name = problem_name[:-5]

    suffix = '' if layer_id is None else '_layer_' + str(layer_id)
    if contact_opt:
        suffix += '_contact'
    save_path = os.path.join(MT_DIR, problem_name + '_MT' + suffix + '.json')

    contact_pairs_coord = opt_data["contact_pairs_coord"]
    nstatus = opt_data["nstatus"]
    xs = opt_data["xs"]
    vs = opt_data["vs"]

    line_endpt_pairs = list_to_pairs(listify(line_pts))
    coupler_endpt_pairs = list_to_pairs(listify(coupler_endpts))


    data = {
        'line_pt_pairs' : line_endpt_pairs,
        'contact_pairs_coord' : contact_pairs_coord,
        'coupler_pt_pairs' : coupler_endpt_pairs,
        'nstatus' : nstatus,
        'opt_line_pos': [[x[0], x[1], x[2]] for x in xs],
        'opt_line_orient': [[v[0], v[1], v[2]] for v in vs],
        'opt_edges_coord': opt_data['edges_coord'],
        'opt_parameters': smilp_parameters
    }


    with open(save_path, 'w') as f:
        json.dump(data, f, indent=0)

    cprint('data saved to {}'.format(save_path), 'green')

def convex_combination(x1, x2, w=0.5):
    assert 0 <= w and w <= 1
    return (1 - w) * np.array(x1) + (w * np.array(x2))

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

def compute_reciprocal_contact_pairs(V, E):
    for vid in range(len(V)):
        v_vertices = []
        for e in E:
            if vid in e:
                if e[0] == vid:
                    v_vertices.append(e[1])
                else:
                    v_vertices.append(e[0])

        drts = []
        for vend in v_vertices:
            drts.append(vend - V[vid])

        print(V[vid])





# V, E, FE all coordinates
def remove_vertex_duplication(V, E, FE, Layers, parameters):
    newV = []
    dangling_V = [True for v in V]

    map = {}

    for e in E:
        dangling_V[e[0]] = False
        dangling_V[e[1]] = False

    for id in range(len(V)):
        if dangling_V[id]:
            continue

        v = V[id]
        duplicate = False
        for nid in range(len(newV)):
            nv = newV[nid]
            if np.linalg.norm(v - nv) < parameters["duplicate_vertices_tol"]:
                duplicate = True
                break

        if duplicate:
            map[id] = nid
        else:
            map[id] = len(newV)
            newV.append(v)

    newE = []
    newFE = []
    for e in E:
        if map[e[0]] != map[e[1]]:
            new_e = [map[e[0]], map[e[1]]]
            newE.append(new_e)
            if e in FE:
                newFE.append(new_e)

    newLayers = []
    for layer in Layers:
        new_layer = []
        for e in layer:
            if map[e[0]] != map[e[1]]:
                new_e = [map[e[0]], map[e[1]]]
                new_layer.append(new_e)
        newLayers.append(new_layer)

    return [newV, newE, newFE, newLayers]

def dist_between_two_bars(nA, nB, xA, xB):
    """
    :param nA: the direction of bar A
    :param nB: the direction of bar B
    :param xA: the position of bar A
    :param xB: the position of bar B
    :return: the distance between the two bars assuming they are not parallel
    """
    n = jnp.cross(nA, nB)
    n /= jnp.linalg.norm(n)
    return jnp.dot(xA - xB, n)

def parameter_tA(nA, nB, xA, xB):
    """
    :param nA: the direction of bar A
    :param nB: the direction of bar B
    :param xA: the position of bar A
    :param xB: the position of bar B
    :return: tA, the intersection point which are parameterized on bar A
    """
    Mat = jnp.array([[nA.dot(nA), -nA.dot(nB)],
                    [nB.dot(nA), -nB.dot(nB)]])
    b = jnp.array([(xB - xA).dot(nA), (xB - xA).dot(nB)])
    T = jnp.linalg.inv(Mat) @ b
    return T[0]

def parameter_tB(nA, nB, xA, xB):
    """
    :param nA: the direction of bar A
    :param nB: the direction of bar B
    :param xA: the position of bar A
    :param xB: the position of bar B
    :return: tB, the intersection point which are parameterized on bar A
    """
    Mat = jnp.array([[nA.dot(nA), -nA.dot(nB)],
                    [nB.dot(nA), -nB.dot(nB)]])
    b = jnp.array([(xB - xA).dot(nA), (xB - xA).dot(nB)])
    T = jnp.linalg.inv(Mat) @ b
    return T[1]

def dot(x, y):
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def add(x, y):
    return [x[0] + y[0], x[1] + y[1], x[2] + y[2]]


def diff(x, y):
    return [x[0] - y[0], x[1] - y[1], x[2] - y[2]]