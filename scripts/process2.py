from typing import List
import threading
import time
import sys
import json
import itertools
import sys
import re
import math
import copy
#import scipy.optimize
import random
import matplotlib.pyplot as plt
import numpy as np


import pandas

from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd




def convert_logged_measurement(val):
    return val / 1000.0

def extract_types(msg_iter, types):
    rounds = set()

    by_round_and_dev = {}

    for t in types:
        by_round_and_dev[t] = {}

    for (ts, d, msg) in msg_iter:

        if 'type' not in msg:
            continue

        if msg['type'] not in types:
            continue

        round = msg['round']
        rounds.add(round)

        by_round_and_dev[msg['type']][(d, round)] = msg

    rounds = list(rounds)
    rounds.sort()

    return tuple([rounds] + [by_round_and_dev[t] for t in types])



def extract_measurements(msg_iter):
    rounds, raw_measurements, drift_estimations = extract_types(msg_iter, ['raw_measurements', 'drift_estimation'])

    # put all messages into (d, round) sets

    print(raw_measurements.keys())

    for r in rounds:
        for d in devs:
            for (a, da) in enumerate(devs):
                for (b, db) in enumerate(devs):
                    if b <= a:
                        continue
                    pi = pair_index(a,b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a,b)
                    record['dist'] = get_dist(dev_positions[da], dev_positions[db])

                    msg = raw_measurements.get((d, r), None)

                    record['estimated_tof'] = None
                    record['round_dur'] = None
                    record['response_dur'] = None

                    if msg is not None and pi < len(msg['measurements']) and msg['measurements'][pi] is not None:
                        record['round_dur'] = msg['measurements'][pi][0]
                        record['response_dur'] = msg['measurements'][pi][1]

                        if msg['measurements'][pi][2] is not None:
                            record['estimated_tof'] = convert_ts_to_m(convert_logged_measurement(msg['measurements'][pi][2]))

                    msg = drift_estimations.get((d, r), None)
                    record['own_dur_a'] = None
                    record['other_dur_a'] = None
                    record['relative_drift_a'] = None
                    record['relative_drift_a_ci'] = None
                    record['own_dur_b'] = None
                    record['other_dur_b'] = None
                    record['relative_drift_b'] = None
                    record['relative_drift_b_ci'] = None

                    if msg is not None:

                        if 'carrierintegrators' in msg:
                            if msg['carrierintegrators'][a] != 0:
                                record['relative_drift_a_ci'] = ci_to_rd(msg['carrierintegrators'][a])
                            if msg['carrierintegrators'][b] != 0:
                                record['relative_drift_b_ci'] = ci_to_rd(msg['carrierintegrators'][b])

                        if msg['durations'][a] is not None:
                            record['own_dur_a'] = msg['durations'][a][0]
                            record['other_dur_a'] = msg['durations'][a][1]
                            if record['own_dur_a'] != 0 and record['other_dur_a'] != 0:
                                record['relative_drift_a'] = float(record['own_dur_a']) / float(record['other_dur_a'])

                        if msg['durations'][b] is not None:
                            record['own_dur_b'] = msg['durations'][b][0]
                            record['other_dur_b'] = msg['durations'][b][1]
                            if record['own_dur_b'] != 0 and record['other_dur_b'] != 0:
                                record['relative_drift_b'] = float(record['own_dur_b']) / float(record['other_dur_b'])

                    record['calculated_tof'] = None
                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'], record['response_dur']]:
                        record['calculated_tof'] = convert_ts_to_m(record['relative_drift_a'] *record['round_dur'] - record['relative_drift_b'] * record['response_dur'])*0.5

                    yield record


def extract_estimations(msg_iter):
    rounds, estimations = extract_types(msg_iter, ['estimation'])

    records = []
    for r in rounds:
        for d in devs:
            for (a, da) in enumerate(devs):
                for (b, db) in enumerate(devs):
                    if b <= a:
                        continue

                    pi = pair_index(a,b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a,b)
                    record['dist'] = get_dist(dev_positions[da], dev_positions[db])

                    msg = estimations.get((d, r), None)

                    if msg is not None and pi < len(msg['mean_measurements']):
                        record['mean_measurement'] = convert_ts_to_m(convert_logged_measurement(msg['mean_measurements'][pi]))
                        record['est_distance_uncalibrated'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_uncalibrated'][pi]))
                        record['est_distance_factory'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_factory_delays'][pi]))
                        record['est_distance_calibrated'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_estimated_delays'][pi]))

                        if 'tofs_from_filtered_estimated_delays' in msg:
                            record['est_distance_calibrated_filtered'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_filtered_estimated_delays'][pi]))
                    else:
                        record['mean_measurement'] = None
                        record['est_distance_uncalibrated'] = None
                        record['est_distance_factory'] = None
                        record['est_distance_calibrated'] = None
                        record['est_distance_calibrated_filtered'] = None

                    yield record


#from testbed.lille import name, dev_positions, parse_messages_from_lines, devs
from testbed.trento_a import name, dev_positions, parse_messages_from_lines, devs

LOG_FILE = "job"

LOG_PATH= "data/{}/{}.log".format(name, LOG_FILE)
EXPORT_PATH= "export/{}/{}".format(name, LOG_FILE)

import os
os.makedirs(EXPORT_PATH, exist_ok = True)

def export_measurements():

    import eval
    eval.load_plot_defaults()

    with open(LOG_PATH) as f:

        src_dev = devs[1]

        meas_df = pandas.DataFrame.from_records(extract_measurements(parse_messages_from_lines(f, src_dev=src_dev)))

        # we first plot the number of measurements for all pairs

        meas_df = meas_df[(meas_df['device'] == src_dev)]

        meas_df = meas_df[(meas_df['round'] <= 1000)]

        meas_df['offset'] = (meas_df['estimated_tof'] - meas_df['dist'])*100

        for i in range(0, len(devs)):
            d0 = devs[i]

            for other in range(0, i):
                if i == other:
                    continue
                df = meas_df
                df = df[(df['initiator'] == other) & ((df['responder'] == i))]

                ax = df.plot(kind='scatter', x='round', y='relative_drift_a', color='b', label='Init Rel. Drift',
                             alpha=0.5, figsize=(20, 10))
                ax = df.plot(ax=ax, kind='scatter', x='round', y='relative_drift_a_ci', color='c', label='Init Rel. Drift (CI)', alpha=0.5)
                ax = df.plot(ax=ax, kind='scatter', x='round', y='relative_drift_b', color='r', label='Resp Rel. Drift', alpha=0.5)
                ax = df.plot(ax=ax, kind='scatter', x='round', y='relative_drift_b_ci', color='y', label='Resp Rel. Drift (CI)', alpha=0.5)
                print("Saving {}-{}".format(i, other))
                plt.title("Rel. Drifts {}-{}".format(i, other))
                plt.savefig("{}/measurements_rel_drifts_{}-{}.pdf".format(EXPORT_PATH, i, other))
                plt.close()

            for other in range(0, i-1):
                df = meas_df
                df = df[(df['initiator'] == other) & ((df['responder'] == i))]

                df_b = meas_df[(meas_df['initiator'] == other+1) & ((meas_df['responder'] == i))]

                ax = df.plot(kind='scatter', x='round', y='offset', color='C0',  label='{}-{}'.format(i+1, other+1), alpha=0.5, figsize=(5, 4), edgecolors='none')
                ax = df_b.plot(ax=ax, kind='scatter', x='round', y='offset', color='C1', label='{}-{}'.format(i+1, other+2), alpha=0.5, edgecolors='none')

                #plt.axhline(y=get_dist(dev_positions[d0], dev_positions[devs[other]]), color='b', linestyle='-')

                #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_single', color='b', label='C (32 bit)')
                #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_int_10', color='b', label='Python (Integer)')
                print("Saving {}-{}".format(i, other))

                plt.grid(color='lightgray', linestyle='dashed')

                ax.set_ylabel('Offset [cm]')
                ax.set_xlabel('Round')

                plt.gcf().set_size_inches(5.0, 4.5)
                plt.tight_layout()

                #plt.title("Scatter {}-{}".format(i, other))
                plt.savefig("{}/measurements_scatter_{}-{}.pdf".format(EXPORT_PATH, i, other))

                plt.close()
                #plt.show()

        # plot the amount of measurements
        df = meas_df
        df = df[['pair', 'estimated_tof']]
        df = df.groupby(['pair']).agg('count')
        df.plot.bar(y=['estimated_tof'])
        plt.savefig("{}/measurements_count.pdf".format(EXPORT_PATH))
        plt.close()

def export_estimations():

    with open(LOG_PATH) as f:

        src_dev = devs[0]

        est_df = pandas.DataFrame.from_records(extract_estimations(parse_messages_from_lines(f, src_dev=src_dev)))

        round = est_df['round'].max()
        est_df = est_df[(est_df['round'] == round) & (est_df['mean_measurement'].notna())]

        est_df['err_uncalibrated'] = est_df['est_distance_uncalibrated'] - est_df['dist']
        est_df['err_factory'] = est_df['est_distance_factory'] - est_df['dist']
        est_df['err_calibrated'] = est_df['est_distance_calibrated'] - est_df['dist']
        est_df['err_calibrated_filtered'] = est_df['est_distance_calibrated_filtered'] - est_df['dist']

        est_df['abs_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.abs)
        est_df['abs_err_factory'] = est_df['err_factory'].apply(np.abs)
        est_df['abs_err_calibrated'] = est_df['err_calibrated'].apply(np.abs)
        est_df['abs_err_calibrated_filtered'] = est_df['err_calibrated_filtered'].apply(np.abs)

        est_df['squared_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.square)
        est_df['squared_err_factory'] = est_df['err_factory'].apply(np.square)
        est_df['squared_err_calibrated'] = est_df['err_calibrated'].apply(np.square)
        est_df['squared_err_calibrated_filtered'] = est_df['err_calibrated_filtered'].apply(np.square)

        df = est_df.sort_values(by='err_uncalibrated')
        # df.plot.bar(x='pair',y=['dist', 'est_distance_uncalibrated', 'est_distance_factory', 'est_distance_calibrated'])
        df.plot.bar(x='pair', y=['err_uncalibrated', 'err_factory', 'err_calibrated'])
        plt.savefig("{}/est_err_uncalibrated_sorted.pdf".format(EXPORT_PATH))
        plt.close()


        res = df[['squared_err_uncalibrated', 'squared_err_factory', 'squared_err_calibrated', 'squared_err_calibrated_filtered']].aggregate(func=['mean'])
        res = res[['squared_err_uncalibrated', 'squared_err_factory', 'squared_err_calibrated', 'squared_err_calibrated_filtered']].apply(np.sqrt)

        ax = res.plot.bar()
        for container in ax.containers:
            ax.bar_label(container)

        plt.savefig("{}/est_aggr_rmse.pdf".format(EXPORT_PATH))
        plt.close()

        df = est_df[['abs_err_uncalibrated', 'abs_err_factory', 'abs_err_calibrated', 'abs_err_calibrated_filtered']]
        ax = df.hist(alpha=0.33, bins=20)

        plt.savefig("{}/est_aggr_hist.pdf".format(EXPORT_PATH))
        plt.close()

        df = est_df[['initiator', 'responder', 'abs_err_uncalibrated', 'abs_err_factory', 'abs_err_calibrated', 'abs_err_calibrated_filtered']]

        df_init = df
        df_resp = df

        df_init = df_init.rename(columns={'initiator': 'id'})[['id', 'abs_err_uncalibrated']]
        df_resp = df_resp.rename(columns={'responder': 'id'})[['id', 'abs_err_uncalibrated']]

        df_both = pandas.concat([df_init, df_resp])

        res = df_both.groupby('id').aggregate(func=['median', 'mean', 'std', 'max'])
        ax = res.plot.bar()
        for container in ax.containers:
            ax.bar_label(container)

        plt.savefig("{}/est_pair_errors.pdf".format(EXPORT_PATH))
        plt.close()

if __name__ == "__main__":

    export_estimations()
    export_measurements()



    #
    # # first plot the mean measurements for all connections from the first device
    # # df = df_m[(df_m['device'] == d0) & (df_m['initiator'] <= 4) & (df_m['responder'] == 0) & (df_m['mean_measurement'].notna())]
    # # df = df[['mean_measurement', 'round', 'initiator']]
    # # df = df.pivot(index='round', columns= 'initiator', values='mean_measurement')
    # # df.plot()
    # # plt.show()
    #
    #