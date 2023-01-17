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

SPEED_OF_LIGHT = 299792458.0
MIN_CALIBRATION_DIST = 0.0

DEFAULT_ANT_DELAY = 16450

# The sum of both tx and rx delays, is measured for round times and subtracted from delay times
COMBINED_DEFAULT_DELAY_S = 513.0/1.0e9


DWT_FREQ_OFFSET_MULTIPLIER = (998.4e6/2.0/1024.0/131072.0)
DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 = (-1.0e6/6489.6e6)

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

def get_dbm_adjustment(dbm):
    return 0 # not used atm
    dbm = int(dbm)
    if dbm in DBM_TWR_DIST_ADJUST:
        return DBM_TWR_DIST_ADJUST[dbm]
    else:
        assert dbm-1 in DBM_TWR_DIST_ADJUST and dbm+1 in DBM_TWR_DIST_ADJUST
        return (DBM_TWR_DIST_ADJUST[dbm-1]*0.5+DBM_TWR_DIST_ADJUST[dbm+1]*0.5)

def adjust_dist_by_rssi(dist, rssi):
    return dist-get_dbm_adjustment(rssi)

def handle_overflow_in_sec(ordered_vals):
    OVERFLOW_OFFSET = 0xFFFFFFFFFF
    OVERFLOW_OFFSET_SEC = convert_ts_to_sec(OVERFLOW_OFFSET)

    for i in range(len(ordered_vals)-1):
        if ordered_vals[i] > ordered_vals[i+1]:
            for j in range(i+1, len(ordered_vals)):
                ordered_vals[j] += OVERFLOW_OFFSET_SEC

    # for i in range(len(ordered_vals)):
    #     for j in range(len(ordered_vals)):
    #         assert i >= j or ordered_vals[i] < ordered_vals[j]

    return ordered_vals

def msg_get_rx_ts(tx_msg, rx_msg):
    for msg in tx_msg['rx']:
        if msg['addr'] == rx_msg['tx']['addr'] and msg['sn'] == rx_msg['tx']['sn']:
            return msg['ts']
    return None

def convert_ts_to_sec(ts):
    return (ts*15650.0)*1e-15

def convert_sec_to_ts(sec):
    ts = (sec / 1e-15) / 15650.0
    assert convert_ts_to_sec(ts)-sec < 0.00001
    return ts


# this is a recursive generator... might not be that easy to comprehend! (TODO: add more comments)
def extract_message_sequences(source, source_history, sequence, dependency_list, include_all_index=0) :
    # we go through the sequence and extract the messages
    # if the current part of the sequence is our source, then

    if len(sequence) == 0:
        yield []    # return empty sequence too
        return

    # collect all possible partial sequences (which is just an empty list for the start and then lists with a possible first message)
    partial_sequences = extract_message_sequences(source, source_history, sequence[:-1], dependency_list, include_all_index)
    addr = sequence[-1]


    for ps in partial_sequences:
        # ps is a list of messages (i.e. the sequence)
        # we now search the source_history for a matching entry that also satisfies all wanted includes
        # HANDLE addr == source case also!
        for m in source_history:
            if addr != m['tx']['addr']:
                continue

            assert (addr == source and m['type'] == 'tx') or (addr != source and m['type'] == 'rx')

            # now check that all includes were found
            dependencies = dependency_list[len(ps)]

            all_includes_found = True
            for i in dependencies:
                dep = ps[i]
                # check that this is included (cannot be if the same sender!)
                if m['tx']['addr'] != dep['tx']['addr'] and msg_get_rx_ts(m, dep) is None:
                    all_includes_found = False
            if all_includes_found:
                yield ps + [m]
                if len(ps) > include_all_index:
                    break

def calc_twr(a, b, history):
    assert a != b

    for seq in extract_message_sequences(a, history, [a, b, a], [[], [0], [1]], 0):
        (poll_msg, response_msg, final_msg) = seq

        poll_tx_ts_a = convert_ts_to_sec(poll_msg['tx']['ts'])
        poll_rx_ts_b = convert_ts_to_sec(msg_get_rx_ts(response_msg, poll_msg))

        response_tx_ts_b = convert_ts_to_sec(response_msg['tx']['ts'])
        response_rx_ts_a = convert_ts_to_sec(msg_get_rx_ts(final_msg, response_msg))

        final_tx_ts_a = convert_ts_to_sec(final_msg['tx']['ts'])

        assert None not in [poll_tx_ts_a, poll_rx_ts_b, response_tx_ts_b, response_rx_ts_a, final_tx_ts_a]

        [poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a] = handle_overflow_in_sec([poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a])
        [poll_rx_ts_b, response_tx_ts_b] = handle_overflow_in_sec([poll_rx_ts_b, response_tx_ts_b])

        assert poll_tx_ts_a < response_rx_ts_a < final_tx_ts_a
        assert poll_rx_ts_b < response_tx_ts_b


        print(response_msg['carrierintegrator'])
        cio = response_msg['carrierintegrator'] * (DWT_FREQ_OFFSET_MULTIPLIER * DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6)
        clock_ratio_ka_kb = 1.0 - cio

        round_a = response_rx_ts_a - poll_tx_ts_a
        delay_b = response_tx_ts_b - poll_rx_ts_b

        tof_dc_a = (round_a-clock_ratio_ka_kb*delay_b) / 2.0

        yield tof_dc_a * SPEED_OF_LIGHT

def calc_alt_twr(a, b, history):

    assert a != b

    for seq in extract_message_sequences(a, history, [a, b, a, b], [[], [0], [1], [2], [3]]):
        (poll_msg, response_msg, final_msg, data_msg) = seq

        poll_tx_ts_a = convert_ts_to_sec(poll_msg['tx']['ts'])
        poll_rx_ts_b = convert_ts_to_sec(msg_get_rx_ts(response_msg, poll_msg))

        response_tx_ts_b = convert_ts_to_sec(response_msg['tx']['ts'])
        response_rx_ts_a = convert_ts_to_sec(msg_get_rx_ts(final_msg, response_msg))

        final_tx_ts_a = convert_ts_to_sec(final_msg['tx']['ts'])
        final_rx_ts_b = convert_ts_to_sec(msg_get_rx_ts(data_msg, final_msg))

        assert None not in [poll_tx_ts_a, poll_rx_ts_b, response_tx_ts_b, response_rx_ts_a, final_tx_ts_a, final_rx_ts_b]

        [poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a] = handle_overflow_in_sec([poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a])
        [poll_rx_ts_b, response_tx_ts_b, final_rx_ts_b] = handle_overflow_in_sec([poll_rx_ts_b, response_tx_ts_b, final_rx_ts_b])

        assert poll_tx_ts_a < response_rx_ts_a < final_tx_ts_a
        assert poll_rx_ts_b < response_tx_ts_b < final_rx_ts_b

        round_a = response_rx_ts_a - poll_tx_ts_a
        delay_a = final_tx_ts_a - response_rx_ts_a

        round_b = final_rx_ts_b - response_tx_ts_b
        delay_b = response_tx_ts_b - poll_rx_ts_b

        clock_ratio_ka_kb = (round_a + delay_a) / (delay_b + round_b)

        print(response_msg['carrierintegrator'])
        cio = response_msg['carrierintegrator'] * (
                    DWT_FREQ_OFFSET_MULTIPLIER * DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6)
        estimated_clock_ratio_ka_kb = 1.0 - cio

        print(clock_ratio_ka_kb, estimated_clock_ratio_ka_kb)

        #print(clock_ratio_ka_kb)


        combined_delay_a = COMBINED_DEFAULT_DELAY_S
        combined_delay_b = COMBINED_DEFAULT_DELAY_S

        # we apply the combined delay for b after the conversion so we have everything in source time
        # this is not guaranteed to be better but we at least have consistent error ;)
        tof_dc_a = (round_a-clock_ratio_ka_kb*delay_b) / 2.0 - (combined_delay_a+combined_delay_b) / 2.0
        dist = tof_dc_a * SPEED_OF_LIGHT
        #print(a, b, dist)
        rssi = response_msg['rssi']

        yield adjust_dist_by_rssi(dist, rssi)



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

def plot_carrier_integrator_values(msg_by_dev, dev_mapping, inv_dev_mapping, n='dwm1001-1'):

    collected_ts = {}
    collected_ci = {}

    for d in dev_mapping:
        if d != n:
            collected_ts[d] = []
            collected_ci[d] = []

    min_ts = None
    for msg in msg_by_dev[n]:
        if msg['type'] == 'rx':

            if min_ts is None:
                min_ts = msg['_log_ts']
            collected_ts[inv_dev_mapping[msg['tx']['addr']]].append(msg['_log_ts'] - min_ts)
            cio = msg['carrierintegrator'] #* (DWT_FREQ_OFFSET_MULTIPLIER * DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6)
            collected_ci[inv_dev_mapping[msg['tx']['addr']]].append(cio)

    for d in collected_ci:
        plt.plot(collected_ts[d], collected_ci[d], label=d)
        plt.text(collected_ts[d][-1], collected_ci[d][-1], d)

    # plot lines
    plt.legend()
    plt.show()

    print("Carrier integrator values")
    print(collected_ci)

# def time_scaler(dev_msg_iter):
#
#     num_overflows = {}
#     last_tx_ts = {}
#
#     def correct_ts(d, ts):
#         assert d in num_overflows
#         return num_overflows[d] * 0xFFFFFFFFFF +ts
#
#     # we expect that each device sends at least once per overflow (i.e. 17 seconds in the decawave case)
#     for (d, msg) in dev_msg_iter:
#         if d not in num_overflows:
#             num_overflows[d] = 0
#             last_tx_ts[d] = None
#
#         if msg['type'] == 'tx':
#             if last_tx_ts[d] is not None and msg['tx']['ts'] < last_tx_ts[d]:
#                 num_overflows[d] += 1
#
#             last_tx_ts[d] = msg['tx']['ts']
#
#         # we correct all timestamps for the actual transmission and the reception timestamps (just in the tx msgs)
#         print(msg)
#         if msg['type'] == 'tx':
#             msg['tx']['ts'] = correct_ts(d, msg['tx']['ts'])
#             for rx_msg in msg['rx']:
#                 rx_msg['ts'] = correct_ts(d, rx_msg['ts'])
#
#         print(msg)
#         yield d, msg


# def apply_antenna_offset_to_iter(dev_msg_iter, rx_offsets, tx_offsets):
#     for (d, msg) in dev_msg_iter:
#         # note that the source dev is not d in the case of receptions!
#         source_dev = msg['tx']['addr']
#         txo = tx_offsets.get(source_dev, 0)
#         rxo = rx_offsets.get(source_dev, 0)
#
#         msg['tx']['ts'] += txo - DEFAULT_ANT_DELAY
#
#         for r in msg['rx']:
#             r['ts'] -= rxo - DEFAULT_ANT_DELAY
#
#         yield d, msg

def apply_antenna_offset_to_map(msg_by_dev, rx_offsets, tx_offsets):
    msg_by_dev = copy.deepcopy(msg_by_dev)  # we do not want to mess with the original data
    for d in msg_by_dev:
        for msg in msg_by_dev[d]:
            # note that the source dev is not d in the case of receptions!
            source_dev = msg['tx']['addr']
            txo = tx_offsets.get(source_dev, 0)
            rxo = rx_offsets.get(source_dev, 0)

            # if the antenna has additional delay, the transmission timestep is too early
            msg['tx']['ts'] += txo - DEFAULT_ANT_DELAY

            # if the antenna has additional delay, the reception timestep is too late
            for r in msg['rx']:
                r['ts'] -= rxo-DEFAULT_ANT_DELAY
    return msg_by_dev

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


for (a, da) in enumerate(dev_positions):
    for (b, db) in enumerate(dev_positions):
        if a > b:
            print((a,b, round(get_dist(da, db), 4)))


if __name__ == "__main__":
    with open("data/serial_output_lille.log") as f:
        dev_msg_iter = parse_messages_from_lines(f)

        msg_by_dev = {}
        dev_mapping = {}

        # we just initialize everything here to
        for d in dev_positions:
            dev_mapping[d] = None
            msg_by_dev[d] = []

        # we now iterate through the messages and set the address as well
        for (ts, d, msg) in dev_msg_iter:

            if msg['type'] == 'tx':
                # We check that we did not generate the same device twice

                assert dev_mapping[d] is None or dev_mapping[d] == msg['tx']['addr']
                dev_mapping[d] = msg['tx']['addr']
            msg_by_dev[d].append(msg)

        # we check that not two devices generated the same mac addr
        # for e.g. 14 devices the probability is around 0.15% so not too high (birthday paradox) ;)
        dev_mapping_set = set(dev_mapping.values())
        assert len(dev_mapping) == len(dev_mapping)

        inv_dev_mapping = {v: k for k, v in dev_mapping.items()}

        print("Device mapping:")
        print(dev_mapping)

        #plot_carrier_integrator_values(msg_by_dev, dev_mapping, inv_dev_mapping, 'dwm1001-1')


        # we first checkout the carrier integrator values for a single node:
        dev_a = 'dwm1001-1'
        dev_b = 'dwm1001-14'


        print("DIST: {}".format(get_dist(dev_a, dev_b)))

        twrs = calc_alt_twr(dev_mapping[dev_a], dev_mapping[dev_b], msg_by_dev[dev_a])

        ys = np.array(list(twrs))

        plt.hist(ys, density=True, bins=50)

        plt.legend()
        plt.show()

        print("TWR Values")
        print(sorted(ys))




        exit()









        left_anchors = []
        right_anchors = []
        tags = []

        dev_dist_mapping_ref = {}

        for a in dev_positions:
            dev_dist_mapping_ref[a] = {}
            for b in dev_positions:
                dev_dist_mapping_ref[a][b] = math.dist(dev_positions[a], dev_positions[b])

        #print(dev_dist_mapping_ref)

        def estimate_distance_matrix(dev_mapping, msg_by_dev):
            m = {}
            for a in dev_mapping:
                m[a] = {}
                for b in dev_mapping:
                    if a == b:
                        m[a][b] = 0
                    else:
                        xs = list(calc_alt_twr(dev_mapping[a], dev_mapping[b], msg_by_dev[a]))
                        if len(xs) == 0:
                            m[a][b] = None
                        else:
                            xs.sort()
                            quantile_len = round(len(xs) * 0.05)
                            if quantile_len > 0:
                                xs = xs[quantile_len:-quantile_len]
                            m[a][b] = sum(xs)/len(xs)
            return m  # estimate the distances by all devices

        def calculate_calibration_loss(dev_mapping, dev_dist_ref, msg_by_dev, rx_offsets, tx_offsets):

            msg_by_dev = apply_antenna_offset_to_map(msg_by_dev, rx_offsets, tx_offsets)
            dist_est = estimate_distance_matrix(dev_mapping, msg_by_dev)

            err_sum = 0.0
            err_num = 0
            for a in dev_mapping:
                for b in dev_mapping:
                    if a != b and dev_dist_ref[a][b] > MIN_CALIBRATION_DIST and dist_est[a][b] is not None:
                        err = abs(dev_dist_ref[a][b] - dist_est[a][b])
                        #err_sum += err  #MAE
                        err_sum += err*err  #MSE
                        err_num += 1
            err_num = max(err_num, 1)
            print(err_sum / err_num)
            return err_sum / err_num


        def calibrate(dev_mapping, dev_dist_ref, msg_by_dev, rand=False):
            devs = []
            for d in dev_mapping:
                devs.append(dev_mapping[d])
            if rand:
                random.shuffle(devs)

            def params_to_offsets(params):
                assert len(params) == 2*len(devs)

                # we try to approximate half of the total delay according to decawave, (split to 44% for tx and 56% for rx delays)

                rx_offsets = {}
                tx_offsets = {}
                for (i, d) in enumerate(devs):
                    rx_offsets[d] = params[2*i]
                    tx_offsets[d] = params[2*i+1]

                #print(rx_offsets, tx_offsets)
                return (rx_offsets, tx_offsets)

            def loss(params):
                print(params)
                (rx_offsets, tx_offsets) = params_to_offsets(params)
                return calculate_calibration_loss(dev_mapping, dev_dist_ref, msg_by_dev, rx_offsets, tx_offsets)

            avg_delay_ts = 16450 #convert_sec_to_ts(515.0*1e-9)
            delay_span_ts = convert_sec_to_ts(4.0*1e-9)
            bounds = (avg_delay_ts-delay_span_ts, avg_delay_ts+delay_span_ts)
            #print("bound", bounds)
            if rand:
                initial_delays = [random.uniform(bounds[0], bounds[1]) for p in range(len(devs)*2)]
            else:
                initial_delays = [avg_delay_ts for p in range(len(devs)*2)]

            res = scipy.optimize.minimize(loss, initial_delays, bounds=[bounds]*len(devs)*2, method='Powell')
            return params_to_offsets(res.x)

        # 0.003
        #(rx_offsets, tx_offsets) = ({'0xBB1E': 14406.927603513546, '0x9042': 14399.393921091114, '0xE621': 14502.832007397377, '0x853C': 14491.195700323082}, {'0xBB1E': 18336.08967719906, '0x9042': 18326.501354115964, '0xE621': 18458.149827596662, '0x853C': 18443.33998222938})
        # 0.0033095516400284625
        #(rx_offsets, tx_offsets) = ({'0xBB1E': 14406.929284833172, '0x9042': 14399.394679103347, '0xE621': 14502.831814957344, '0x853C': 14491.194413552084}, {'0xBB1E': 18336.0918170604, '0x9042': 18326.502318858806, '0xE621': 18458.149582672984, '0x853C': 18443.33834452084})



        (rx_offsets, tx_offsets) = ({'0xBB1E': 16293.04339946, '0x9042': 16449.97652584, '0xE621': 16275.88954547, '0x853C': 16449.99418732}, {'0xBB1E': 16510.98191186, '0x9042': 16450.00129933, '0xE621': 16484.52522657, '0x853C': 16450.01300476})

        #for i in range(20):
        (rx_offsets, tx_offsets) = calibrate(dev_mapping, dev_dist_mapping_ref, msg_by_dev)
        print(calculate_calibration_loss(dev_mapping, dev_dist_mapping_ref, msg_by_dev, rx_offsets, tx_offsets), rx_offsets, tx_offsets)

        offset_msg_by_dev = apply_antenna_offset_to_map(msg_by_dev, rx_offsets, tx_offsets)

        for t in tags:
            for l in left_anchors:
                for r in right_anchors:
                    if t in dev_mapping and l in dev_mapping and r in dev_mapping:
                        xs = list(calc_tdoa_alt_twr(dev_mapping[t], dev_mapping[l], dev_mapping[r], offset_msg_by_dev[t]))
                        if len(xs) == 0:
                            print(t,l,r,None)
                        else:
                            xs.sort()
                            #print(xs)
                            quantile_len = math.ceil(len(xs) * 0.05)
                            xs_90 = xs[quantile_len:-quantile_len]
                            print(t, l, r, "100% Mean: {}, 90% Mean: {}, Median: {}".format(sum(xs) / len(xs),
                                                                                         sum(xs_90) / len(xs_90),
                                                                                         xs[int(len(xs) / 2)]))

        #(rx_offsets, tx_offsets) = calibrate(dev_mapping, dev_dist_mapping_ref, msg_by_dev)
        #print((rx_offsets, tx_offsets))


        #print(dev_dist_mapping_ref)
        #print(estimate_distance_matrix(dev_mapping, msg_by_dev))

        #print()
        # 0.03 [ 1.56954142e+02 -2.44577925e-02  1.74093177e+02 -9.30810992e-03 -6.10009614e+01 -1.79911868e-02 -3.44278714e+01  9.03475992e-03]
        # ({'0xBB1E': 13.984941018139757, '0x9042': 25.250549514444568, '0xE621': 22.863062611145047, '0x853C': 19.158230417479277}, {'0xBB1E': -11.763788632923006, '0x9042': -42.294306855383724, '0xE621': 7.274973733565174, '0x853C': -31.737094588865645})

        # if we have a tx message, we try to compute the TDoA values with everyone else
        #if msg['type'] == 'tx':# and d== '/dev/tty.usbmodem0007601202631':

        # we now loop through every device and determine its TDoA value to every other device pairs!
        # (at least based on the messages that this particular device received!)
        #
        # methods = {
        #     'alt': calc_tdoa_alt_twr,
        #     'ss': calc_tdoa_ss_twr
        # }
        #
        # vals_by_method = {}
        # for m in methods:
        #     vals_by_method[m] = {}
        #     for (l, (a, b)) in itertools.product(msg_by_dev, itertools.combinations(msg_by_dev, 2)):
        #         if l in [a, b]:
        #             continue    # we do not want to TDoA ourselves ;)
        #         #methods[m](l, a, b, msg_by_dev[l])
        #
        #         vals_by_method[m][(l, a, b)] = []
        #         for i in range(len(msg_by_dev[l])):
        #             vals_by_method[m][(l, a, b)].append(
        #                 methods[m](l, a, b, msg_by_dev[l][i:])
        #             )
        #print(vals_by_method)