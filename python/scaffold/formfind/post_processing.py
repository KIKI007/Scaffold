import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scaffold.formfind.util import compute_closest_t_between_lines, write_json_smilp
from scaffold.geometry import ScaffoldModel

def compute_scaffold_model(ct, opt_data, opt_parameters, save=False, format=None):
    # * save layer result
    [resultV, resultCouplerV] = post_processing(opt_data, opt_parameters)

    for v in resultV:
        v += ct

    for v in resultCouplerV:
        v += ct

    contact_ts = []
    contact_pairs_coord = opt_data["contact_pairs_coord"]
    edges_coord = opt_data["edges_coord"]

    adj = []

    for (e1, e2) in contact_pairs_coord:

        e1_ind = edges_coord.index(e1)
        e2_ind = edges_coord.index(e2)
        adj.append((e1_ind, e2_ind))

        t1, t2 = compute_closest_t_between_lines(resultV[e1_ind * 2],
                                                 resultV[e1_ind * 2 + 1],
                                                 resultV[e2_ind * 2],
                                                 resultV[e2_ind * 2 + 1])
        contact_ts.extend([t1, t2])

    # save to file
    if save:
        write_json_smilp(opt_data,
                         resultV,
                         resultCouplerV,
                         opt_parameters,
                         problem_name = format["name"],
                         contact_opt=format["complete"],
                         layer_id=format["id"])

    # save to model

    line = []
    for id in range(int(len(resultV) / 2)):
        p0 = resultV[id * 2]
        p1 = resultV[id * 2 + 1]
        line.append([p0, p1])

    contact_points = []
    for id in range(int(len(resultCouplerV) / 2)):
        p0 = resultCouplerV[id * 2]
        p1 = resultCouplerV[id * 2 + 1]
        contact_points.append([p0, p1])

    model = ScaffoldModel(line, adj, contact_points, opt_data["bar_radius"])
    model.load_default_coupler()
    return model

def post_processing(opt_data, parameters):
    """
    :param opt_data: a set contains
           "xs", the optimal positions of the infinite length bars
           "vs", the optimal orientations of the infinite length bars
           "contact_pairs", the pairs of bars that are in contact
    :param stock_bar_lengths:
    :return [resultV, resultCouplerV]: every two consecutive points represent the end points of finite length bars.
    """
    contact_pairs_coord = opt_data["contact_pairs_coord"]
    stock_bar_lengths = parameters["bar_available_lengths"]
    edges_coord = opt_data["edges_coord"]
    vertices_coord = opt_data["vertices_coord"]
    fixed_edges_coord = opt_data["fixed_edges_coord"]

    xs = opt_data["xs"]
    vs = opt_data["vs"]

    # Endpoints' locations (returned)
    resultV = []

    resultCouplerV = []

    # the arc length of endpoints (returned)
    resultT = [[1E6, -1E6] for eit in range(len(edges_coord))]

    # compute intersection points between adjacent bars
    # the intersection points will be
    for (e1, e2) in contact_pairs_coord:
        if e1 not in edges_coord or e2 not in edges_coord:
            continue

        e1_ind = edges_coord.index(e1)
        e2_ind = edges_coord.index(e2)

        v1 = vs[e1_ind]
        v2 = vs[e2_ind]

        x1 = xs[e1_ind]
        x2 = xs[e2_ind]

        Mat = np.array([[v1.dot(v1), -v1.dot(v2)],
                        [v2.dot(v1), -v2.dot(v2)]])
        b = np.array([(x2 - x1).dot(v1), (x2 - x1).dot(v2)])

        if np.linalg.norm(np.cross(v1, v2)) < 1E-6:
            T = [np.dot((x2 - x1), v1) / 2, np.dot(x1 - x2, v2) / 2]
            print(T)
        else:
            T = np.linalg.inv(Mat) @ b

        e_inds = [e1_ind, e2_ind]
        for kd in range(0, 2):
            ind = e_inds[kd]
            resultT[ind][0] = min(T[kd], resultT[ind][0])
            resultT[ind][1] = max(T[kd], resultT[ind][1])
            resultCouplerV.append(xs[ind] + T[kd] * vs[ind])

    clamp_extension_length = parameters["clamp_extension_length"]

    for eit in range(len(edges_coord)):
        v = vs[eit]
        x = xs[eit]
        if resultT[eit][0] == 1E6:
            resultT[eit][1] = resultT[eit][0] = 0

        curr_bar_length = (resultT[eit][1] - resultT[eit][0]) + clamp_extension_length * 2

        if curr_bar_length > stock_bar_lengths[-1]:
            print("bar ", eit, " ({}),".format(curr_bar_length), "is longer than any stock bar")

        additional_bar_length = 0
        for jd in range(0, len(stock_bar_lengths)):
            if jd == 0 and stock_bar_lengths[jd] >= curr_bar_length:
                additional_bar_length = stock_bar_lengths[jd] - curr_bar_length
                break
            elif stock_bar_lengths[jd] >= curr_bar_length and curr_bar_length > stock_bar_lengths[jd - 1]:
                additional_bar_length = stock_bar_lengths[jd] - curr_bar_length
                break
            else:
                pass

        ts_before_opt = []
        for jd in range(2):
            pt = vertices_coord[edges_coord[eit][jd]]
            t = np.dot((pt - x), v)
            ts_before_opt.append(t)

        # Create a new gurobi model
        m = gp.Model("bilinear")
        ts = m.addVars(2, lb=[-GRB.INFINITY, -GRB.INFINITY], ub=[GRB.INFINITY, GRB.INFINITY])
        stock_bar_vars = m.addVars(len(stock_bar_lengths), vtype=GRB.BINARY)

        sum_expr = 0
        length_expr = 0
        for jd in range(len(stock_bar_lengths)):
            length_expr += stock_bar_lengths[jd] * stock_bar_vars[jd]
            sum_expr += stock_bar_vars[jd]

        m.addConstr(sum_expr == 1)
        m.addConstr(-ts[0] + ts[1] == length_expr)
        m.addConstr(ts[0] <= resultT[eit][0] - clamp_extension_length)
        m.addConstr(ts[1] >= resultT[eit][1] + clamp_extension_length)

        if eit in fixed_edges_coord:
            m.addConstr(ts[0] <= ts_before_opt[0] - clamp_extension_length)
            m.addConstr(ts[1] >= ts_before_opt[1] + clamp_extension_length)

        m.setObjective((ts[0] - ts_before_opt[0]) * (ts[0] - ts_before_opt[0]) + (ts[1] - ts_before_opt[1]) * (ts[1] - ts_before_opt[1]), GRB.MINIMIZE)

        m.Params.OutputFlag = 0
        m.optimize()

        if m.SolCount == 0:
            resultT[eit][1] += additional_bar_length / np.linalg.norm(v) / 2
            resultT[eit][0] -= additional_bar_length / np.linalg.norm(v) / 2
        else:
            resultT[eit][0] = ts[0].X
            resultT[eit][1] = ts[1].X

        for jd in range(0, 2):
            resultV.append(resultT[eit][jd] * v + x)

    return [resultV, resultCouplerV]