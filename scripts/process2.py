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

SPEED_OF_LIGHT = 299792458.0
MIN_CALIBRATION_DIST = 0.0

DEFAULT_ANT_DELAY = 16450

# The sum of both tx and rx delays, is measured for round times and subtracted from delay times
COMBINED_DEFAULT_DELAY_S = 513.0/1.0e9

DWT_FREQ_OFFSET_MULTIPLIER = (998.4e6/2.0/1024.0/131072.0)
DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 = (-1.0e6/6489.6e6)

METER_PER_DWT_TS = ((SPEED_OF_LIGHT * 1.0E-15) * 15650.0)

DBM_TWR_DIST_ADJUST = {
    -61: -11,
    -63: -10.5,
    -65: -10.0,
    -67: -9.3,
    -69: -8.2,
    -71: -6.9,
    -73: -5.1,
    -75: -2.7,
    -77: 0.0,
    -79: 2.1,
    -81: 3.5,
    -83: 4.2,
    -85: 4.9,
    -87: 6.2,
    -89: 7.1,
    -91: 7.6,
    -93: 8.1
}
#
# print(METER_PER_DWT_TS)
#
# for k in range(-61, -94, -1):
#     if k in DBM_TWR_DIST_ADJUST:
#         cm = DBM_TWR_DIST_ADJUST[k]
#     else:
#         cm = (DBM_TWR_DIST_ADJUST[k-1] + DBM_TWR_DIST_ADJUST[k+1]) / 2.0
#
#     m = cm/100.0
#     ts_correction = round(m / METER_PER_DWT_TS)
#     print("{ts_correction}, // {rssi}dBm ({cm} cm)".format(ts_correction=ts_correction, rssi=k, cm=cm))
#



def convert_ts_to_sec(ts):
    return (ts*15650.0)*1e-15

def convert_sec_to_ts(sec):
    ts = (sec / 1e-15) / 15650.0
    assert convert_ts_to_sec(ts)-sec < 0.00001
    return ts

def convert_ts_to_m(ts):
    return convert_ts_to_sec(ts)*SPEED_OF_LIGHT


def parse_messages_from_lines(line_it):
    for line in line_it:
        if line.strip() == "":
            continue
        try:
            # 1670158055.359572;dwm1001-2;{ "type": "rx", "carrierintegrator": 1434, "rssi": -81, "tx": {"addr": "0xBC48", "sn": 0, "ts": 186691872256}, "rx": [{"addr": "0x471A", "sn": 0, "ts": 185512854420}]}
            log_ts, dev, json_str = line.split(';', 3)
            log_ts = float(log_ts)

            try:
                msg = json.loads(json_str)
                msg['_log_ts'] = log_ts
                yield (log_ts, dev, msg)
            except json.decoder.JSONDecodeError:
                #print(json_str)
                pass
        except ValueError:
            pass

dev_positions = {
        'dwm1001-1': (23.31,	0.26,	7.55),
        'dwm1001-2': (24.51,	0.26,	8.96),
        'dwm1001-3': (26.91,	0.26,	8.96),
        'dwm1001-4': (28.11,	0.26,	7.55),
        'dwm1001-5': (25.11,	1.1,	9.51),
        'dwm1001-6': (27.51,	1.1,	9.51),
        'dwm1001-7': (25.11,	3.5,	9.51),
        'dwm1001-8': (27.51,	3.5,	9.51),
        'dwm1001-9': (26.31,	5.9,	9.51),
        'dwm1001-10': (25.11,	7.1,	9.51),
        'dwm1001-11': (27.51,	7.1,	9.51),
        'dwm1001-12': (24.51,	9.39,	7.52),
        'dwm1001-13': (25.71,	9.39,	9.22),
        'dwm1001-14': (26.91,	9.39,	7.52)
    }

def get_dist(a, b):
    pos_a = np.array(dev_positions[a])
    pos_b = np.array(dev_positions[b])
    return np.linalg.norm(pos_a - pos_b)


def pair_index(a, b):
    if a > b:
        return int((a * (a - 1)) / 2 + b)
    else:
        return pair_index(b, a)


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
                    record['dist'] = get_dist(da, db)

                    msg = raw_measurements.get((d, r), None)

                    record['estimated_tof'] = None
                    record['round_dur'] = None
                    record['response_dur'] = None

                    if msg is not None and pi < len(msg['measurements']) and msg['measurements'][pi] is not None:
                        record['round_dur'] = msg['measurements'][pi][0]
                        record['response_dur'] = msg['measurements'][pi][1]

                        if msg['measurements'][pi][2] is not None:
                            record['estimated_tof'] = convert_logged_measurement(msg['measurements'][pi][2])*METER_PER_DWT_TS

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
                        record['calculated_tof'] = (record['relative_drift_a'] *record['round_dur'] - record['relative_drift_b'] * record['response_dur'])*0.5*METER_PER_DWT_TS
                        record['calculated_tof_single'] = (
                                                              np.subtract(
                                                                np.multiply(
                                                                    np.divide(
                                                                        np.float32(record['own_dur_a']),
                                                                        np.float32(record['other_dur_a'])),
                                                                    np.float32(record['round_dur'])),
                                                                  np.multiply(
                                                                      np.divide(
                                                                          np.float32(record['own_dur_b']),
                                                                          np.float32(record['other_dur_b'])),
                                                                      np.float32(record['response_dur'])))
                                                          )*0.5*METER_PER_DWT_TS
                        # record['calculated_tof_single_v2'] = np.subtract(
                        #                                           np.divide(
                        #                                                 np.multiply(np.float32(record['own_dur_a']),
                        #                                                             np.float32(record['round_dur'])),
                        #                                                 np.float32(record['other_dur_a'])
                        #                                           ),
                        #                                           np.divide(
                        #                                               np.multiply(np.float32(record['own_dur_b']),
                        #                                                           np.float32(record['response_dur'])),
                        #                                               np.float32(record['other_dur_b'])
                        #                                           )
                        #                                   ) * 0.5 * METER_PER_DWT_TS

                        accurator = math.pow(2, 10)
                        record['calculated_tof_int_10'] = np.float64(np.add(
                            np.subtract(np.longlong(record['round_dur']*accurator), np.longlong(record['response_dur']*accurator)),
                            np.subtract(
                                np.floor_divide(
                                    np.multiply(
                                        np.subtract(np.longlong(record['own_dur_a']*accurator), np.longlong(record['other_dur_a']*accurator)),
                                        np.longlong(record['round_dur'])
                                    ),
                                    np.longlong(record['other_dur_a'])
                                ),
                                np.floor_divide(
                                    np.multiply(
                                        np.subtract(np.longlong(record['own_dur_b']*accurator), np.longlong(record['other_dur_b']*accurator)),
                                        np.longlong(record['response_dur'])
                                    ),
                                    np.longlong(record['other_dur_b'])
                                )
                            )
                        )) * (1.0/accurator) * 0.5 * METER_PER_DWT_TS
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
                    record['dist'] = get_dist(da, db)

                    msg = estimations.get((d, r), None)

                    if msg is not None and pi < len(msg['mean_measurements']):
                        record['mean_measurement'] = convert_logged_measurement(msg['mean_measurements'][pi])*METER_PER_DWT_TS
                        record['est_distance_uncalibrated'] = convert_logged_measurement(msg['tofs_uncalibrated'][pi])*METER_PER_DWT_TS
                        record['est_distance_factory'] = convert_logged_measurement(msg['tofs_from_factory_delays'][pi])*METER_PER_DWT_TS
                        record['est_distance_calibrated'] = convert_logged_measurement(msg['tofs_from_estimated_delays'][pi])*METER_PER_DWT_TS
                    else:
                        record['mean_measurement'] = None
                        record['est_distance_uncalibrated'] = None
                        record['est_distance_factory'] = None
                        record['est_distance_calibrated'] = None

                    yield record


if __name__ == "__main__":
    with open("data/serial_output_with_measurements_new.log") as f:

        meas_df = pandas.DataFrame.from_records(extract_measurements(parse_messages_from_lines(f)))

        # we first plot the number of measurements for all pairs

        df = meas_df[(meas_df['device'] == 'dwm1001-1')]
        df = df.sort_values(by='initiator')

        df = df[['pair',  'initiator', 'responder', 'round', 'estimated_tof', 'calculated_tof', 'calculated_tof_single', 'calculated_tof_int_10']]

        df = df[(df['initiator'] == 3) & ((df['responder'] == 9))]

        df = df[['round', 'estimated_tof', 'calculated_tof', 'calculated_tof_single', 'calculated_tof_int_10']]

        ax = df.plot( kind='scatter', x='round', y='calculated_tof', color='b', label='Python (64 bit)', alpha=0.5)
        ax = df.plot(ax=ax, kind='scatter', x='round', y='estimated_tof', color='r', label='C', alpha=0.5)
        #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_single', color='b', label='C (32 bit)')
        #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_int_10', color='b', label='Python (Integer)')

        plt.show()

        exit()

        #print(df)

        #df = df[['pair', 'measurement']]
        #df = df.groupby(['pair']).agg('count')
        #df.plot.bar( y=['measurement'])
        #plt.show()



        # first plot the mean measurements for all connections from the first device
        # df = df_m[(df_m['device'] == 'dwm1001-1') & (df_m['initiator'] <= 4) & (df_m['responder'] == 0) & (df_m['mean_measurement'].notna())]
        # df = df[['mean_measurement', 'round', 'initiator']]
        # df = df.pivot(index='round', columns= 'initiator', values='mean_measurement')
        # df.plot()
        # plt.show()



        # we now iterate through the messages and set the address as well
        est_df = pandas.DataFrame.from_records(extract_estimations(parse_messages_from_lines(f)))

        round = est_df['round'].max()

        df = est_df[(est_df['round'] == round) & (est_df['device'] == 'dwm1001-1') & (est_df['mean_measurement'].notna())]

        df['err_uncalibrated'] = df['est_distance_uncalibrated'] - df['dist']
        df['err_factory'] = df['est_distance_factory'] - df['dist']
        df['err_calibrated'] = df['est_distance_calibrated'] - df['dist']

        df['abs_err_uncalibrated'] = df['err_uncalibrated'].apply(np.abs)
        df['abs_err_factory'] = df['err_factory'].apply(np.abs)
        df['abs_err_calibrated'] = df['err_calibrated'].apply(np.abs)

        df = df.sort_values(by='err_calibrated')
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
        df = df[['abs_err_uncalibrated', 'abs_err_factory', 'abs_err_calibrated']]

        res= df.aggregate(func=['median', 'mean', 'std'])
        ax = res.plot.bar()
        for container in ax.containers:
            ax.bar_label(container)

        plt.show()
        #q_up = df.quantile(q=0.95)
        #q_low = df.quantile(q=0.05)
        #print(res, q_low, q_up)