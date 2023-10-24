import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base import pair_index
from scipy.spatial import distance
from scipy.optimize import least_squares

SIDE_LENGTH = 10.0

NODES_POSITIONS = np.asarray([
    (0.4*SIDE_LENGTH, 0.6*SIDE_LENGTH),
    (0.4*SIDE_LENGTH, 0.4*SIDE_LENGTH),
    (0.6*SIDE_LENGTH, 0.6*SIDE_LENGTH),
])

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




def compute_rmse(true_positions, anchors, active_devices, passive_devices, tof_measurements, tdoa_measurements, use_tdoa_for_active=False):

    anchor_positions = {}
    for a in anchors:
        anchor_positions[a] = true_positions[a]

    # todo use_tdoa_for_active=True not working as of now!

    # we first position all of the active nodes
    real_dists = [np.linalg.norm(p - n) for n in active_devices]
    meas_twr_dists =

    def act_err_fun(est_positions)
