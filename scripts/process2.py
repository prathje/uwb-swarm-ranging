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

# TODO: CHOOSE THE TESTBED HERE!
from testbed.trento_b import dev_positions, parse_messages_from_lines, devs

from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts




def convert_logged_measurement(val):
    return val / 1000.0

def extract_types(msg_iter, types):
    rounds = set()

    by_round_and_dev = {}

    for t in types:
        by_round_and_dev[t] = {}

    for (ts, d, msg) in msg_iter:

        if d != devs[0]:
            continue

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
    devices = dev_positions.keys()

    for r in rounds:
        for d in devices:
            for (a, da) in enumerate(dev_positions):
                for (b, db) in enumerate(dev_positions):
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
                    record['relative_drift_a_single'] = None
                    record['own_dur_b'] = None
                    record['other_dur_b'] = None
                    record['relative_drift_b'] = None
                    record['relative_drift_b_single'] = None

                    if msg is not None and msg['durations'][a] is not None:
                        record['own_dur_a'] = msg['durations'][a][0]
                        record['other_dur_a'] = msg['durations'][a][1]
                        if record['own_dur_a'] != 0 and record['other_dur_a'] != 0:
                            record['relative_drift_a'] = float(record['own_dur_a']) / float(record['other_dur_a'])
                            record['relative_drift_a_single'] = np.divide(np.float32(record['own_dur_a']), np.float32(record['other_dur_a']))

                    if msg is not None and msg['durations'][b] is not None:
                        record['own_dur_b'] = msg['durations'][b][0]
                        record['other_dur_b'] = msg['durations'][b][1]
                        if record['own_dur_b'] != 0 and record['other_dur_b'] != 0:
                            record['relative_drift_b'] = float(record['own_dur_b']) / float(record['other_dur_b'])
                            record['relative_drift_b_single'] = np.divide(np.float32(record['own_dur_b']), np.float32(record['other_dur_b']))

                    record['calculated_tof'] = None
                    record['calculated_tof_single'] = None
                    record['calculated_tof_single_v2'] = None
                    record['calculated_tof_single_v3'] = None
                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'], record['response_dur']]:
                        record['calculated_tof'] = convert_ts_to_m(record['relative_drift_a'] *record['round_dur'] - record['relative_drift_b'] * record['response_dur'])*0.5
                        # record['calculated_tof_single'] = (
                        #                                       np.subtract(
                        #                                         np.multiply(
                        #                                             np.divide(
                        #                                                 np.float32(record['own_dur_a']),
                        #                                                 np.float32(record['other_dur_a'])),
                        #                                             np.float32(record['round_dur'])),
                        #                                           np.multiply(
                        #                                               np.divide(
                        #                                                   np.float32(record['own_dur_b']),
                        #                                                   np.float32(record['other_dur_b'])),
                        #                                               np.float32(record['response_dur'])))
                        #                                   )*0.5*METER_PER_DWT_TS
                        # # record['calculated_tof_single_v2'] = np.subtract(
                        # #                                           np.divide(
                        # #                                                 np.multiply(np.float32(record['own_dur_a']),
                        # #                                                             np.float32(record['round_dur'])),
                        # #                                                 np.float32(record['other_dur_a'])
                        # #                                           ),
                        # #                                           np.divide(
                        # #                                               np.multiply(np.float32(record['own_dur_b']),
                        # #                                                           np.float32(record['response_dur'])),
                        # #                                               np.float32(record['other_dur_b'])
                        # #                                           )
                        # #                                   ) * 0.5 * METER_PER_DWT_TS
                        #
                        # accurator = math.pow(2, 10)
                        # record['calculated_tof_int_10'] = np.float64(np.add(
                        #     np.subtract(np.longlong(record['round_dur']*accurator), np.longlong(record['response_dur']*accurator)),
                        #     np.subtract(
                        #         np.floor_divide(
                        #             np.multiply(
                        #                 np.subtract(np.longlong(record['own_dur_a']*accurator), np.longlong(record['other_dur_a']*accurator)),
                        #                 np.longlong(record['round_dur'])
                        #             ),
                        #             np.longlong(record['other_dur_a'])
                        #         ),
                        #         np.floor_divide(
                        #             np.multiply(
                        #                 np.subtract(np.longlong(record['own_dur_b']*accurator), np.longlong(record['other_dur_b']*accurator)),
                        #                 np.longlong(record['response_dur'])
                        #             ),
                        #             np.longlong(record['other_dur_b'])
                        #         )
                        #     )
                        # )) * (1.0/accurator) * 0.5 * METER_PER_DWT_TS
                    #print(record['estimated_tof'], record['calculated_tof'])
                    yield record


def extract_estimations(msg_iter):
    rounds, estimations = extract_types(msg_iter, ['estimation'])

    devices = dev_positions.keys()

    records = []
    for r in rounds:
        for d in devices:
            for (a, da) in enumerate(dev_positions):
                for (b, db) in enumerate(dev_positions):
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
                    else:
                        record['mean_measurement'] = None
                        record['est_distance_uncalibrated'] = None
                        record['est_distance_factory'] = None
                        record['est_distance_calibrated'] = None

                    yield record


if __name__ == "__main__":
    with open("data/trento_b/job_busy_wait.log") as f:


        d0 = devs[0]
        d3 = devs[3]
        d9 = devs[9]

        meas_df = pandas.DataFrame.from_records(extract_measurements(parse_messages_from_lines(f)))

        # we first plot the number of measurements for all pairs

        for i in range(0, len(devs)):
            d0 = devs[i]

            for other in range(0, i):
                df = meas_df[(meas_df['device'] == devs[0])]
                df = df.sort_values(by='initiator')

                df = df[['pair',  'initiator', 'responder', 'round', 'estimated_tof', 'calculated_tof', ]]

                df = df[(df['initiator'] == other) & ((df['responder'] == i))]

                df = df[['round', 'estimated_tof', 'calculated_tof']]

                ax = df.plot(kind='scatter', x='round', y='calculated_tof', color='b', label='Python (64 bit)', alpha=0.5, figsize=(20, 10), ylim=(0, 12))
                ax = df.plot(ax=ax, kind='scatter', x='round', y='estimated_tof', color='r', label='C', alpha=0.5)

                plt.axhline(y=get_dist(dev_positions[d0], dev_positions[devs[other]]), color='r', linestyle='-')

                #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_single', color='b', label='C (32 bit)')
                #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_int_10', color='b', label='Python (Integer)')
                print("Saving {}-{}".format(i, other))
                plt.title("{}-{}".format(i, other))
                plt.savefig("export/{}-{}.pdf".format(i, other))
                #plt.show()


        #print(df)

        #df = df[['pair', 'measurement']]
        #df = df.groupby(['pair']).agg('count')
        #df.plot.bar( y=['measurement'])
        #plt.show()



        # first plot the mean measurements for all connections from the first device
        # df = df_m[(df_m['device'] == d0) & (df_m['initiator'] <= 4) & (df_m['responder'] == 0) & (df_m['mean_measurement'].notna())]
        # df = df[['mean_measurement', 'round', 'initiator']]
        # df = df.pivot(index='round', columns= 'initiator', values='mean_measurement')
        # df.plot()
        # plt.show()



        # we now iterate through the messages and set the address as well
        est_df = pandas.DataFrame.from_records(extract_estimations(parse_messages_from_lines(f)))

        round = est_df['round'].max()

        df = est_df[(est_df['round'] == round) & (est_df['device'] == d0) & (est_df['mean_measurement'].notna())]

        df['err_uncalibrated'] = df['est_distance_uncalibrated'] - df['dist']
        df['err_factory'] = df['est_distance_factory'] - df['dist']
        df['err_calibrated'] = df['est_distance_calibrated'] - df['dist']

        df['abs_err_uncalibrated'] = df['err_uncalibrated'].apply(np.abs)
        df['abs_err_factory'] = df['err_factory'].apply(np.abs)
        df['abs_err_calibrated'] = df['err_calibrated'].apply(np.abs)

        df = df.sort_values(by='err_uncalibrated')
        #df.plot.bar(x='pair',y=['dist', 'est_distance_uncalibrated', 'est_distance_factory', 'est_distance_calibrated'])
        df.plot.bar(x='pair', y=['err_uncalibrated', 'err_factory', 'err_calibrated'])
        plt.show()
        #
        # plt.clf()
        # df[(est_df['initiator'] == 1)].plot.bar(x='pair', y=['err_uncalibrated', 'err_factory', 'err_calibrated'])
        # plt.show()


        df['squared_err_uncalibrated'] = df['err_uncalibrated'].apply(np.square)
        df['squared_err_factory'] = df['err_factory'].apply(np.square)
        df['squared_err_calibrated'] = df['err_calibrated'].apply(np.square)

        #df = df[['squared_err_uncalibrated', 'squared_err_factory', 'squared_err_calibrated']]
        df = df[['initiator', 'responder', 'abs_err_uncalibrated', 'abs_err_factory', 'abs_err_calibrated']]

        df_init = df
        df_resp = df

        df_init = df_init.rename(columns={'initiator': 'id'})[['id', 'abs_err_uncalibrated']]
        df_resp = df_resp.rename(columns={'responder': 'id'})[['id', 'abs_err_uncalibrated']]

        df_both = pandas.concat([df_init, df_resp])

        print(df_both)


        res = df_both.groupby('id').aggregate(func=['median', 'mean', 'std', 'max'])
        ax = res.plot.bar()
        for container in ax.containers:
            ax.bar_label(container)

        plt.show()
        #q_up = df.quantile(q=0.95)
        #q_low = df.quantile(q=0.05)
        #print(res, q_low, q_up)