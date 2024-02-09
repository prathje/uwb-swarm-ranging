import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import LinAlgError

from base import pair_index
from scipy.spatial import distance
from scipy.optimize import least_squares

SIDE_LENGTH = 10.0

p = 0.4
p2 = 0.2

NODES_POSITIONS = np.asarray([
    (p*SIDE_LENGTH, p2*SIDE_LENGTH),
    (p*SIDE_LENGTH, (1.0-p2)*SIDE_LENGTH),
    ((1.0-p)*SIDE_LENGTH, (1.0-p2)*SIDE_LENGTH),
    ((1.0-p)*SIDE_LENGTH, p2*SIDE_LENGTH)
])

# rands = np.random.rand(8)
# print(rands)
# NODES_POSITIONS = np.asarray([
#     (rands[0]*SIDE_LENGTH, rands[1]*SIDE_LENGTH),
#     (rands[2]*SIDE_LENGTH, rands[3]*SIDE_LENGTH),
#     (rands[4]*SIDE_LENGTH, rands[5]*SIDE_LENGTH),
#     (rands[6]*SIDE_LENGTH, rands[7]*SIDE_LENGTH)
# ])


EPSILON = 1e-5


NODE_PAIRS = []

for a in range(len(NODES_POSITIONS)):
    for b in range(len(NODES_POSITIONS)):
        if a < b:
            NODE_PAIRS.append((NODES_POSITIONS[a], NODES_POSITIONS[b]))

def sim_single_tof_positioning(p, meas_std):
    p = np.asarray(p)
    real_dists = [np.linalg.norm(p - n) for n in NODES_POSITIONS]
    meas_dists = np.random.normal(loc=np.asarray(real_dists), scale=meas_std)

    def err_fun(x):
        dists = np.asarray([np.linalg.norm(x - n) for n in NODES_POSITIONS])
        return meas_dists-dists
    res = least_squares(err_fun, p, bounds=(-1, SIDE_LENGTH+1))
    return np.linalg.norm(p - res.x)

def sim_single_tdoa_positioning(p, meas_std):
    p = np.asarray(p)

    pos_pairs = NODE_PAIRS

    real_diffs = [np.linalg.norm(p - na) - np.linalg.norm(p - nb) for (na, nb) in pos_pairs]

    meas_diffs = np.random.normal(loc=np.asarray(real_diffs), scale=meas_std)

    def err_fun(x):
        diffs = np.asarray([np.linalg.norm(x - na) - np.linalg.norm(x - nb) for (na, nb) in pos_pairs])
        return meas_diffs - diffs

    res = least_squares(err_fun, p, bounds=(-1, SIDE_LENGTH+1))
    return np.linalg.norm(p - res.x)



# def tof_squared_residual_derivatives(p, a, m):
#
#
#     dist_a = np.linalg.norm(p - a)
#     if dist_a < 1e-5:
#         dist_a = 1e-5
#
#     dm = 2 * (m - dist_a)
#     dx = -(2 * (p[0] - a[0]) * (m - dist_a))/dist_a
#     dy = -(2 * (p[1] - a[1]) * (m - dist_a))/dist_a
#     #dz = -(2 * (p[2] - a[2]) * (m - dist_a))/dist_a
#
#     return [dm, dx, dy] #, dz]


def tof_residual_derivatives(p, a):

    dist_a = np.linalg.norm(p - a)

    dist_a = np.max([dist_a, EPSILON])

    dm = 1
    dx = -(p[0] - a[0]) / dist_a
    dy = -(p[1] - a[1]) / dist_a
    #dy = -(p[2] - a[2]) / dist_a

    return [dm, dx, dy] #, dz]


#
# def tdoa_squared_residual_derivatives(p, a, b, m):
#
#     dist_a = np.linalg.norm(p - a)
#     dist_b = np.linalg.norm(p - b)
#
#     mult = (-dist_a + dist_b + m)
#
#     dm = 2 * mult
#     dx = 2 * ((p[0] - b[0])/dist_b - (p[0] - a[0])/dist_a) * mult
#     dy = 2 * ((p[1] - b[1])/dist_b - (p[1] - a[1])/dist_a) * mult
#     #dz = 2 * ((p[2] - b[2])/dist_b - (p[2] - a[2])/dist_a) * mult
#     return [dm, dx, dy] #, dz]


def tdoa_residual_derivatives(p, a, b):

    dist_a = np.linalg.norm(p - a)
    dist_b = np.linalg.norm(p - b)

    dm = 1
    dx = ((p[0] - b[0])/dist_b if dist_b != 0.0 else 0.0) - ((p[0] - a[0])/dist_a if dist_a != 0.0 else 0.0 )
    dy = ((p[1] - b[1])/dist_b if dist_b != 0.0 else 0.0) - ((p[1] - a[1])/dist_a if dist_a != 0.0 else 0.0 )


    #dz = (p[2] - b[2])/dist_b - (p[2] - a[2])/dist_a
    return [dm, dx, dy] #, dz]

def sim_single_tof_gdop(p, _meas_std):
    # we actually ignore the measurement error here, as we are only interested in the GDOP

    # we create the matrix of

    H = np.zeros((len(NODES_POSITIONS), 2))

    for i in range(len(NODES_POSITIONS)):
        H[i] = tof_residual_derivatives(p, NODES_POSITIONS[i])[1:]

    Q = np.linalg.inv(np.dot(H.T, H))
    gdop = np.sqrt(np.trace(Q))

    return gdop


def sim_single_tdoa_gdop(p, _meas_std):
    # we actually ignore the measurement error here, as we are only interested in the GDOP

    # we create the matrix of

    H = np.zeros((len(NODE_PAIRS), 2))

    # sample_offsets = [
    #     np.asarray([0.0, 0.0]),
    #     #np.asarray([0.0, 0.0]),
    #     #np.asarray([0.0, EPSILON]),
    #     #np.asarray([EPSILON, EPSILON]),
    #     #np.asarray([EPSILON, 0.0]),
    #     #np.asarray([-EPSILON, 0.0]),
    #     #np.asarray([-EPSILON, -EPSILON]),
    #     #np.asarray([0.0, -EPSILON]),
    # ]

    for (i, (a,b)) in enumerate(NODE_PAIRS):
        # H[i] = np.nanmean(
        #         [tdoa_residual_derivatives(p + off, a, b)[1:] for off in sample_offsets]
        #         , axis=0)
        H[i] = tdoa_residual_derivatives(p, a, b)[1:]

    gdop = np.nan
    try:
        Q = np.linalg.inv(np.matmul(H.T, H))

        trace = np.trace(Q)
        if trace >= 0.0:
            gdop = np.sqrt(trace)
    except LinAlgError as e:
        print("LinAlgError")
        # gdop = np.nan
        # # in this case we are right between two nodes, so the GDOP is infinite
        # # we catch this case and sample in all directions to get a good estimate
        # gdops = []
        # gdops.append(sim_single_tdoa_gdop(p + np.asarray([EPSILON, 0.0]), _meas_std))
        # gdops.append(sim_single_tdoa_gdop(p + np.asarray([-EPSILON, 0.0]), _meas_std))
        # gdops.append(sim_single_tdoa_gdop(p + np.asarray([0.0, EPSILON]), _meas_std))
        # gdops.append(sim_single_tdoa_gdop(p + np.asarray([0.0, -EPSILON]), _meas_std))
        # gdop = np.nanmean(gdops)
        # TODO: we might run into issues when we sample back to the original position

    return gdop

def sim_single_tof_analytical_positioning(p, meas_std):
    p = np.asarray(p)

    real_dists = [np.linalg.norm(p - n) for n in NODES_POSITIONS]


    def calc_position_error_for_meas_err():
        derivs = np.asarray([
            tof_residual_derivatives(p, a, d)[1:] for (a, d) in zip(NODES_POSITIONS, real_dists)
        ])

        derivs = 1.0/derivs

        # derivs are the changes in position for a measurement error of 1
        # hence, we now sample from a normal distribution for n measurements and multiply derivs by that

        errs = []

        for reps in range(1,100):
            meas_errors = np.random.normal(loc=0, scale=meas_std, size=len(NODES_POSITIONS))

            deriv_sample = np.asarray([
                derivs[i] * meas_errors[i] for i in range(len(NODES_POSITIONS))
            ])

            mean_derivs = deriv_sample.mean(axis=0)
            errs.append(np.linalg.norm(mean_derivs))

        errs = np.asarray(errs)
        return np.sqrt(np.mean(errs ** 2))

    loc_err = calc_position_error_for_meas_err()
    return loc_err
    exit()
    # derivs = [
    #     tof_squared_residual_derivatives(p, a, d+meas_std)[0] for (a,d) in zip(NODES_POSITIONS, real_dists)
    # ]



    print(derivs_x, derivs_y)
    exit()

    infs = np.asarray(derivs)

    return np.linalg.norm(infs)
    # we approximate the influence on the least squares method by using the squared residuals derivative functions

def sim_single_tdoa_analytical_positioning(p, meas_std):
    p = np.asarray(p)

    pos_pairs = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if a < b:
                pos_pairs.append((NODES_POSITIONS[a], NODES_POSITIONS[b]))

    real_diffs = [np.linalg.norm(p - na) - np.linalg.norm(p - nb) for (na, nb) in pos_pairs]

    meas_diffs = np.random.normal(loc=np.asarray(real_diffs), scale=meas_std)

    def err_fun(x):
        diffs = np.asarray([np.linalg.norm(x - na) - np.linalg.norm(x - nb) for (na, nb) in pos_pairs])
        return meas_diffs - diffs

    res = least_squares(err_fun, p, bounds=(-1, SIDE_LENGTH+1))
    return np.linalg.norm(p - res.x)

def create_matrix(fun, meas_std, samples_per_side=100, repetitions=10):
    m = np.zeros((samples_per_side, samples_per_side))

    for a in range(samples_per_side):
        for b in range(samples_per_side):
            p = (a * (SIDE_LENGTH/float(samples_per_side)), b * (SIDE_LENGTH/float(samples_per_side)))
            xs = np.asarray([fun(p, meas_std) for x in range(repetitions)])
            m[a,b] = np.sqrt(np.mean(xs**2))
    return m



def position_dist_diff_residual(x, f):
    """
    Compute the negative of the residuals with the exponential model
    for the nonlinear least squares example.
    """
    d = 0
    for meas in measurements:
        first_anchor_coords = meas['first_anchor_coords']
        second_anchor_coords = meas['second_anchor_coords']
        dist_diffs = meas['dists']
        d1 = np.linalg.norm(np.array(x) - np.array(first_anchor_coords))
        d2 = np.linalg.norm(np.array(x) - np.array(second_anchor_coords))
        model_dist_diff = d1 - d2

        for i, observed_dist_diff in enumerate(dist_diffs):
            f[d + i] = model_dist_diff - observed_dist_diff
        d += len(dist_diffs)





def position_dist_diff_partial_derivatives(x, first_anchor_coords, second_anchor_coords, dist1, dist2):
    """
    Compute the partial derivatives of the exponential residual with respect to each parameter in x.
    """
    dx1, dy1, dz1 = 0, 0, 0
    dx2, dy2, dz2 = 0, 0, 0
    epsilon = 1e-5

    if abs(dist1) > epsilon:
        dx1 = (x[0] - first_anchor_coords[0]) / dist1
        dy1 = (x[1] - first_anchor_coords[1]) / dist1
        dz1 = (x[2] - first_anchor_coords[2]) / dist1

    if abs(dist2) > epsilon:
        dx2 = (x[0] - second_anchor_coords[0]) / dist2
        dy2 = (x[1] - second_anchor_coords[1]) / dist2
        dz2 = (x[2] - second_anchor_coords[2]) / dist2

    return np.array([dx1 - dx2, dy1 - dy2, dz1 - dz2])





def position_dist_diff_residual_derivative(x, jacobian):
    """
    Compute the Jacobian matrix of the exponential residual.
    """
    for d, meas in enumerate(measurements):
        first_anchor_coords = meas['first_anchor_coords']
        second_anchor_coords = meas['second_anchor_coords']
        dist_diffs = meas['dists']
        d1 = np.linalg.norm(x - first_anchor_coords)
        d2 = np.linalg.norm(x - second_anchor_coords)

        for i, dist_diff in enumerate(dist_diffs):
            pd_array = position_dist_diff_partial_derivatives(x, first_anchor_coords, second_anchor_coords, d1, d2)
            for j, pd in enumerate(pd_array):
                jacobian[d + i, j] = pd

