from scaffold.formfind.util import *
import gurobipy as gp
from gurobipy import GRB

# pre-compile derivative functions
jit_dist = jit(dist_between_two_bars)
jit_parameter_tA = jit(parameter_tA)
jit_parameter_tB = jit(parameter_tB)
grad_dist = jit(grad(dist_between_two_bars, argnums=(0, 1, 2, 3)))
grad_parameter_tA = jit(grad(parameter_tA, argnums=(0, 1, 2, 3)))
grad_parameter_tB = jit(grad(parameter_tB, argnums=(0, 1, 2, 3)))

def create_bar_distance_constraints(m, vs, xs, Dx_vars, Dv_vars, radius, contact_pairs, parameters):
    for cit in range(0, len(contact_pairs)):
        edgeI = contact_pairs[cit]["edgeI"]
        edgeI_coord = contact_pairs[cit]["edgeI_coord"]
        edgeJ_coord = contact_pairs[cit]["edgeJ_coord"]
        edgeJ = contact_pairs[cit]["edgeJ"]
        edgeI_fix = contact_pairs[cit]["edgeI_fix"]
        edgeJ_fix = contact_pairs[cit]["edgeJ_fix"]
        contact_var = contact_pairs[cit]["contact_var"]
        sign_var = contact_pairs[cit]["sign_var"]

        # lines' attributes at prev iteration
        nA = vs[edgeI]
        nB = vs[edgeJ]
        xA = xs[edgeI]
        xB = xs[edgeJ]

        if parameters["reciprocal"]:
            sign = True
            item = None
            if [edgeI_coord, edgeJ_coord] in parameters["fixed_contact_pairs_coord"]:
                item = [edgeI_coord, edgeJ_coord]
            elif [edgeJ_coord, edgeI_coord] in parameters["fixed_contact_pairs_coord"]:
                sign = not sign
                item = [edgeJ_coord, edgeI_coord]

            if item is not None:
                index = parameters["fixed_contact_pairs_coord"].index(item)
                vn = np.array(parameters["fixed_contact_pairs_normals"][index])
                if vn.dot(np.cross(nA, nB)) < 0:
                    sign = not sign

                m.addConstr(sign_var == sign)

        if [edgeI_coord, edgeJ_coord] in parameters["fixed_contact_pairs_coord"] or \
            [edgeJ_coord, edgeI_coord] in parameters["fixed_contact_pairs_coord"]:
            m.addConstr(contact_var == 1)
        elif parameters["reciprocal"]:
            m.addConstr(contact_var == 0)

        if edgeI_fix and edgeJ_fix:
            if [edgeI_coord, edgeJ_coord] in parameters["fixed_contact_pairs_coord"] or \
                    [edgeJ_coord, edgeI_coord] in parameters["fixed_contact_pairs_coord"]:
                m.addConstr(contact_var == 1)
            else:
                m.addConstr(contact_var == 0)
            continue

        #set constraints
        # if np.linalg.norm(np.cross(nA, nB)) < parameters["contact_ignore_para_tol"]:
        #     m.addConstr(contact_var == 0)

        expr = 0
        if np.linalg.norm(np.cross(nA, nB)) < parameters["line_para_tol"]:
            n1 = np.cross((xA - xB), nA)
            n2 = np.cross(n1, nA)
            if np.linalg.norm(n2) < parameters["line_para_tol"]:
                n2 = np.cross(np.array([1, 0, 0]), nA)
                if np.linalg.norm(n2) < parameters["line_para_tol"]:
                    n2 = np.cross(np.array([0, 1, 0]), nA)

            n2 /= np.linalg.norm(n2)
            # distance
            expr += n2.dot(xA - xB)

            # first order part
            expr += dot(n2, diff(Dx_vars[edgeI], Dx_vars[edgeJ]))
        else:
            grad_dist_coeff = grad_dist(nA, nB, xA, xB)

            # first order part
            expr += dot(grad_dist_coeff[0], Dv_vars[edgeI])
            expr += dot(grad_dist_coeff[1], Dv_vars[edgeJ])
            expr += dot(grad_dist_coeff[2], Dx_vars[edgeI])
            expr += dot(grad_dist_coeff[3], Dx_vars[edgeJ])

            # distance
            expr += dist_between_two_bars(nA, nB, xA, xB)

        M = parameters["bigM"]
        m.addConstr(expr >= radius * 2 - M * (1 - sign_var))
        m.addConstr(expr <= -radius * 2 + M * sign_var)
        m.addConstr(expr <= radius * 2 + M * (1 - contact_var))
        m.addConstr(expr >= -radius * 2 - M * (1 - contact_var))

        ratio = parameters["bar_collision_distance"] / parameters["bar_contact_distance"]
        m.addConstr(expr >= radius * 2 * ratio - M * contact_var - M * (1 - sign_var))
        m.addConstr(expr <= -radius * 2 * ratio + M * contact_var + M * sign_var)

def create_bar_connectivity_constraints(m, V, E, Edge_at_Joint, contact_pairs, parameters):

    contact_vars_at_joint = [[] for v in V]
    num_contact_vars = [[0, 0] for x in E]
    bar_contact_vars = [[0, 0] for x in E]

    for cit in range(len(contact_pairs)):
        edgeI = contact_pairs[cit]["edgeI"]
        edgeJ = contact_pairs[cit]["edgeJ"]

        joint_id = contact_pairs[cit]["joint_id"]
        contact_var = contact_pairs[cit]["contact_var"]

        if np.linalg.norm(V[joint_id] - V[E[edgeI][0]]) < np.linalg.norm(V[joint_id] - V[E[edgeI][1]]):
            bar_contact_vars[edgeI][0] += contact_var
            num_contact_vars[edgeI][0] += 1
        else:
            bar_contact_vars[edgeI][1] += contact_var
            num_contact_vars[edgeI][1] += 1

        if np.linalg.norm(V[joint_id] - V[E[edgeJ][0]]) < np.linalg.norm(V[joint_id] - V[E[edgeJ][1]]):
            bar_contact_vars[edgeJ][0] += contact_var
            num_contact_vars[edgeJ][0] += 1
        else:
            bar_contact_vars[edgeJ][1] += contact_var
            num_contact_vars[edgeJ][1] += 1

        contact_vars_at_joint[joint_id].append([edgeI, edgeJ, contact_var])

    # every joint at least have n - 1 clamps
    for vid in range(len(V)):
        edge_list = Edge_at_Joint[vid]
        num_edge = len(edge_list)
        if num_edge >= 2:
            startInd = edge_list[0]
            ys = []
            for cit in range(len(contact_vars_at_joint[vid])):
                yij = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_edge - 1)
                yji = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_edge - 1)
                contact_var = contact_vars_at_joint[vid][cit][2]
                m.addConstr(yij <= (num_edge - 1) * contact_var)
                m.addConstr(yji <= (num_edge - 1) * contact_var)
                ys.append([yij, yji])

            # start
            expr = 0
            for cit in range(len(contact_vars_at_joint[vid])):
                edgeI = contact_vars_at_joint[vid][cit][0]
                edgeJ = contact_vars_at_joint[vid][cit][1]
                yij = ys[cit][0]
                yji = ys[cit][1]
                if edgeI == startInd:
                    expr += yij
                if edgeJ == startInd:
                    expr += yji

            m.addConstr(expr == num_edge - 1)

            # supply pass
            expr_out = [0 for v in range(num_edge)]
            expr_in = [0 for v in range(num_edge)]

            for cit in range(len(contact_vars_at_joint[vid])):
                edgeI = contact_vars_at_joint[vid][cit][0]
                edgeJ = contact_vars_at_joint[vid][cit][1]
                i0 = edge_list.index(edgeI)
                j0 = edge_list.index(edgeJ)
                yij = ys[cit][0]
                yji = ys[cit][1]
                if edgeI != startInd:
                    expr_out[i0] += yij
                    expr_in[i0] += yji
                if edgeJ != startInd:
                    expr_out[j0] += yji
                    expr_in[j0] += yij

            for id in range(num_edge):
                if edge_list[id] != startInd:
                    m.addConstr(expr_out[id] == expr_in[id] - 1)

   # evert side of a bar should have at least a clamp
    for id in range(len(bar_contact_vars)):
        for jd in range(0, 2):
            m.addConstr(bar_contact_vars[id][jd] >= min(1, num_contact_vars[id][jd]))

def create_clamp_collision_distance_constraints(m, vs, xs, V, E, Dx_vars, Dv_vars, collision_dist, contact_pairs, FE, parameters):

    coupler_edgeIds = [ [] for eit in range(len(E))]
    coupler_ts = [[] for eit in range(len(E))]
    coupler_ts0 = [[] for eit in range(len(E))]
    coupler_contact_vars = [[] for eit in range(len(E))]

    longest_bar_length_in_stock = max(parameters["bar_available_lengths"])
    clamp_extension_length = parameters["clamp_extension_length"]
    longest_bar_length_in_stock = longest_bar_length_in_stock - 2 * clamp_extension_length

    for cit in range(0, len(contact_pairs)):
        edgeI = contact_pairs[cit]["edgeI"]
        edgeJ = contact_pairs[cit]["edgeJ"]
        edgeI_fix = contact_pairs[cit]["edgeI_fix"]
        edgeJ_fix = contact_pairs[cit]["edgeJ_fix"]
        joint_id = contact_pairs[cit]["joint_id"]
        contact_var = contact_pairs[cit]["contact_var"]

        # lines' attributes at prev iteration
        nA = vs[edgeI]
        nB = vs[edgeJ]
        xA = xs[edgeI]
        xB = xs[edgeJ]

        M = parameters["bigM"]
        if np.linalg.norm(np.cross(nA, nB)) >= parameters["contact_ignore_para_tol"]:
            # coupler constraints
            tA = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            tB = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            ts = [tA, tB]
            coupler_ts[edgeI].append(tA)
            coupler_ts[edgeJ].append(tB)
            coupler_contact_vars[edgeI].append(contact_var)
            coupler_contact_vars[edgeJ].append(contact_var)

            coupler_edgeIds[edgeI].append({"id" : edgeJ, "fix": edgeJ_fix})
            coupler_edgeIds[edgeJ].append({"id" : edgeI, "fix": edgeI_fix})

            tA0 = parameter_tA(nA, nB, xA, xB)
            tB0 = parameter_tB(nA, nB, xA, xB)
            tA.Start = tA0
            tB.Start = tB0
            ts0 = [tA0, tB0]
            coupler_ts0[edgeI].append(tA0)
            coupler_ts0[edgeJ].append(tB0)

            grad_tA = grad_parameter_tA(nA, nB, xA, xB)
            grad_tB = grad_parameter_tB(nA, nB, xA, xB)
            grad_ts = [grad_tA, grad_tB]

            for kd in range(2):
                expr = ts[kd]
                expr -= ts0[kd]
                expr -= dot(grad_ts[kd][0], Dv_vars[edgeI])
                expr -= dot(grad_ts[kd][1], Dv_vars[edgeJ])
                expr -= dot(grad_ts[kd][2], Dx_vars[edgeI])
                expr -= dot(grad_ts[kd][3], Dx_vars[edgeJ])
                m.addConstr(expr == 0)

            if edgeI_fix and edgeJ_fix:
                continue

            edgeIndices = [edgeI, edgeJ]
            for kd in range(2):
                edgeId = edgeIndices[kd]
                e0 = V[E[edgeId][0]]
                length = np.linalg.norm(V[joint_id] - e0)
                m.addConstr(ts[kd] >= length - parameters["clamp_t_bnd"] - M * (1 - contact_var))
                m.addConstr(ts[kd] <= length + parameters["clamp_t_bnd"] + M * (1 - contact_var))

    # coupler constraints
    for eit in range(len(E)):
        edgeI = eit
        edgeI_fix = E[eit] in FE
        for id in range(0, len(coupler_ts[eit])):
            edgeJ = coupler_edgeIds[eit][id]["id"]
            edgeJ_fix = coupler_edgeIds[eit][id]["fix"]

            for jd in range(id + 1, len(coupler_ts[eit])):
                edgeK = coupler_edgeIds[eit][jd]["id"]
                edgeK_fix = coupler_edgeIds[eit][jd]["fix"]
                if edgeI_fix and edgeJ_fix and edgeK_fix:
                    continue

                coupler_ts0_j = coupler_ts0[eit][jd]
                coupler_ts0_i = coupler_ts0[eit][id]
                coupler_ts_j = coupler_ts[eit][jd]
                coupler_ts_i = coupler_ts[eit][id]
                contact_var_j = coupler_contact_vars[eit][jd]
                contact_var_i = coupler_contact_vars[eit][id]

                if abs(coupler_ts0_i - coupler_ts0_j) <= parameters["clamp_t_bnd"] * 2:
                    sign_var = m.addVar(vtype = GRB.BINARY)
                    m.addConstr(coupler_ts_j - coupler_ts_i >= collision_dist - M * (1 - contact_var_j) - M * (1 - contact_var_i) - M * sign_var)
                    m.addConstr(coupler_ts_i - coupler_ts_j >= collision_dist - M * (1 - contact_var_j) - M * (1 - contact_var_i) - M * (1 - sign_var))

                #if abs(coupler_ts0_i - coupler_ts0_j) >= 0.5 * longest_bar_length_in_stock:
                m.addConstr(coupler_ts_j - coupler_ts_i <= longest_bar_length_in_stock + 3 * M * (1 - contact_var_j) + 3 * M * (1 - contact_var_i))
                m.addConstr(coupler_ts_i - coupler_ts_j <= longest_bar_length_in_stock + 3 * M * (1 - contact_var_j) + 3 * M * (1 - contact_var_i))

def run_beamopt(E, V, FE, vs, xs, tr_size, parameters):
    """
    :param E: edges' indices
    :param V:  vertices' coordinates
    :param FE: fixed edges' indices
    :param vs: opt lines' orientation vectors
    :param xs: opt lines' origin positions
    :param tr_size: the size of trust region
    :param parameters: optimization related parameters
    :return:
    """

    # assign edges to joints
    Edge_at_Joint = [[] for x in V]
    for vid in range(len(V)):
        for eit in range(len(E)):
            pt = V[vid]
            e0 = V[E[eit][0]]
            e1 = V[E[eit][1]]
            if abs(np.linalg.norm(np.cross(pt - e0, pt - e1))) < parameters["pos_tol"] \
                    and np.dot(pt - e0, pt - e1) < parameters["pos_tol"]:
                Edge_at_Joint[vid].append(eit)

    # Create a new gurobi model
    m = gp.Model("bilinear")

    M = parameters["bigM"]
    Dx_vars = []
    Dv_vars = []
    lb_tr = [-tr_size, -tr_size, -tr_size]
    ub_tr = [tr_size, tr_size, tr_size]
    # Create variables
    for eit in range(0, len(E)):
        if E[eit] in FE:
            # if edge is fixed
            Dx = m.addVars(3, lb=[0, 0, 0], ub=[0, 0, 0])  # location
            Dv = m.addVars(3, lb=[0, 0, 0], ub=[0, 0, 0])  # direction
        else:
            # if edge is not fixed
            Dx = m.addVars(3, lb=lb_tr, ub=ub_tr)  # location
            Dv = m.addVars(3, lb=lb_tr, ub=ub_tr)  # direction

        Dx_vars.append(Dx)
        Dv_vars.append(Dv)

        # lines' position and orientation before opt.
        p0 = V[E[eit][0]]
        p1 = V[E[eit][1]]
        x0 = p0
        v0 = p1 - p0
        v0 /= np.linalg.norm(v0)

        if eit >= len(xs):
            # initialization
            xs.append(x0)
            vs.append(v0)
            v = v0
            x = x0
        else:
            # take prev step values
            v = vs[eit]
            x = xs[eit]

        # linearized constraints of ||v||
        m.addConstr(dot(v, Dv) == 0)

        # shape deviation constraints
        devi_x = diff(add(x, Dx), x0)

        for dim in range(0, 3):
            if E[eit] not in FE:
                m.addConstr(devi_x[dim] <= parameters["pos_devi"])
                m.addConstr(devi_x[dim] >= -parameters["pos_devi"])
        devi_v = dot(add(Dv, v), v0)
        if E[eit] not in FE:
            m.addConstr(devi_v >= math.cos(parameters["orient_devi"]))

    # define variables
    radius = m.addVar(vtype=GRB.CONTINUOUS)
    collision_dist = m.addVar(vtype=GRB.CONTINUOUS)

    # create contact variables
    contact_pairs = []
    for joint_id in range(0, len(Edge_at_Joint)):
        edge_list = Edge_at_Joint[joint_id]

        for id in range(0, len(edge_list)):
            for jd in range(id + 1, len(edge_list)):
                edgeI = edge_list[id]
                edgeJ = edge_list[jd]

                # create vars
                sign_var = m.addVar(vtype=GRB.BINARY)    # sign var
                contact_var = m.addVar(vtype=GRB.BINARY) # contact var

                # contact pairs
                contact_data = {}
                contact_data["edgeI_fix"] = (E[edgeI] in FE)
                contact_data["edgeJ_fix"] = (E[edgeJ] in FE)
                contact_data["edgeI"] = edgeI
                contact_data["edgeJ"] = edgeJ
                contact_data["edgeI_coord"] = E[edgeI]
                contact_data["edgeJ_coord"] = E[edgeJ]
                contact_data["joint_id"] = joint_id
                contact_data["contact_var"] = contact_var
                contact_data["sign_var"] = sign_var
                contact_pairs.append(contact_data)

    # set up constraints
    #

    create_bar_distance_constraints(m, vs, xs, Dx_vars, Dv_vars, radius, contact_pairs, parameters)
    create_bar_connectivity_constraints(m, V, E, Edge_at_Joint, contact_pairs, parameters)
    create_clamp_collision_distance_constraints(m, vs, xs, V, E, Dx_vars, Dv_vars, collision_dist, contact_pairs, FE, parameters)

    # optimization
    clamp_collision_dist = parameters["clamp_collision_dist"]
    bar_contact_distance = parameters["bar_contact_distance"]

    m.addConstr(collision_dist / clamp_collision_dist == radius / bar_contact_distance)
    m.addConstr(radius <= bar_contact_distance)
    m.addConstr(collision_dist <= clamp_collision_dist)
    m.setObjective(radius, GRB.MAXIMIZE)
    m.Params.OutputFlag = parameters["debug_mode"]
    m.Params.TimeLimit = parameters["time_out"]
    m.optimize()

    log = {
        "discrete_vars":m.NumBinVars,
        "continuous_vars":m.NumVars - m.NumBinVars,
        "num_constraints":m.NumConstrs,
        "runtime": m.Runtime,
        "tr_size": tr_size,
        "status": m.SolCount != 0,
        "radius": 0,
    }

    if m.SolCount == 0:
        return [log]

    log.update({"radius": radius.X})


    new_vs = []
    new_xs = []
    for id in range(len(E)):
        new_dv = m.getAttr('X', Dv_vars[id])
        new_dx = m.getAttr('X', Dx_vars[id])
        new_v = np.array([new_dv[0], new_dv[1], new_dv[2]]) + vs[id]
        new_x = np.array([new_dx[0], new_dx[1], new_dx[2]]) + xs[id]
        new_v /= np.linalg.norm(new_v)
        new_vs.append(new_v)
        new_xs.append(new_x)

    new_contact_pairs = []
    for id in range(len(contact_pairs)):
        contact_var = contact_pairs[id]["contact_var"]
        edgeI_coord = contact_pairs[id]["edgeI_coord"]
        edgeJ_coord = contact_pairs[id]["edgeJ_coord"]
        if contact_var.X == 1:
            new_contact_pairs.append([edgeI_coord, edgeJ_coord])

    return [new_vs, new_xs, radius.X, collision_dist.X, new_contact_pairs, log]
