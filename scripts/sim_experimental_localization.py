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




def least_squares_loc(true_positions, anchors, active_devices, passive_devices, tof_measurements, tdoa_measurements, use_cooperative=True, use_tdoa_for_active=False):

    anchor_positions = {}
    for a in anchors:
        anchor_positions[a] = true_positions[a]
        assert a not in active_devices
        assert a not in passive_devices

    # todo use_tdoa_for_active=True not working as of now!

    actual_measurements = []

    # Add measurements of each active device
    for i, active_device in enumerate(active_devices):

        # to every anchor
        for anchor in anchors:
            actual_measurements.append(tof_measurements[(active_device, anchor)])
            #print(active_device, anchor, actual_measurements[-1])

            # also add passive measurements if enabled
            if use_tdoa_for_active:
                for j, other_device in enumerate(active_devices):
                    if active_device != other_device:
                        actual_measurements.append(tdoa_measurements[(active_device, anchor, other_device)])
                        #print(active_device, anchor, other_device, actual_measurements[-1])

        # and every other node
        if use_cooperative:
            for k, other_active_device in enumerate(active_devices):
                if active_device != other_active_device:
                    actual_measurements.append(tof_measurements[(active_device, other_active_device)])
                    #print(active_device, other_active_device, actual_measurements[-1])

                    # also add passive measurements if enabled
                    if use_tdoa_for_active:
                        for j, other_device in enumerate(active_devices):
                            if active_device != other_device and other_active_device != other_device:
                                actual_measurements.append(tdoa_measurements[(active_device, other_active_device, other_device)])
                                #print(active_device, other_device, actual_measurements[-1])

    actual_measurements = np.asarray(actual_measurements)

    def dist(a, b):
        return np.linalg.norm(a-b)

    def err_func(est_active_position_vals):

        est_active_positions = np.reshape(est_active_position_vals, (-1, 2))

        model_meas = []
        for i, active_device in enumerate(active_devices):
            # to every anchor
            for anchor in anchors:
                model_meas.append(dist(est_active_positions[i], anchor_positions[anchor]))
                #print(active_device, anchor, model_meas[-1])

                # also add passive measurements if enabled
                if use_tdoa_for_active:
                    for j, other_device in enumerate(active_devices):
                        if active_device != other_device:
                            model_meas.append(dist(est_active_positions[j], est_active_positions[i]) - dist(est_active_positions[j], anchor_positions[anchor]))
                            #print(active_device, anchor, other_device, model_meas[-1])

            # and every other node
            if use_cooperative:
                for k, other_active_device in enumerate(active_devices):
                    if active_device != other_active_device:
                        model_meas.append(dist(est_active_positions[i], est_active_positions[k]))
                        #print(active_device, other_active_device, model_meas[-1])

                        # also add passive measurements if enabled
                        if use_tdoa_for_active:
                            for j, other_device in enumerate(active_devices):
                                if active_device != other_device and other_active_device != other_device:
                                    model_meas.append(dist(est_active_positions[j], est_active_positions[i]) - dist(est_active_positions[j], est_active_positions[k]))
                                    #print(active_device, other_device, model_meas[-1])

        model_meas = np.asarray(model_meas)

        #print(model_meas)
        #print(actual_measurements)
        #print(model_meas-actual_measurements)

        return model_meas-actual_measurements

    true_anchor_position_list = [true_positions[k] for k in anchor_positions]
    p_mean = np.asarray(true_anchor_position_list).mean(axis=0)
    p_mean = np.reshape(p_mean, (-1, 2))
    p_initial = np.repeat(p_mean, len(active_devices), axis=0)

    res = least_squares(err_func, p_initial.flatten())

    est_positions = np.reshape(res.x, (-1, 2))
    true_position_list = np.asarray([true_positions[k] for k in active_devices])
    active_errs = np.linalg.norm(true_position_list - est_positions, axis=1)

    #for i, k in enumerate(true_position_list):
    #    print("err", np.linalg.norm(true_position_list[i] - est_positions[i]))

    return active_errs, []