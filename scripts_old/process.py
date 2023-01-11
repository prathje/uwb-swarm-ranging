from typing import List
from serial import Serial
import threading
import time
import sys
import json
import itertools
import sys
import re
import math
import copy
import scipy.optimize
import random

SPEED_OF_LIGHT = 299792458.0
MIN_CALIBRATION_DIST = 0.0

DEFAULT_ANT_DELAY = 16450

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




def calc_twr(a, b, history):
    assert a != b

    for seq in extract_message_sequences(a, history, [a, b, a], [[], [0], [1]], 2):
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

        clock_ratio_ka_kb = (1.0+response_msg['clock_ratio_offset']) # ka/kb ? or do we need to negate that? :D

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
        #print(clock_ratio_ka_kb)

        tof_dc_a = (round_a-clock_ratio_ka_kb*delay_b) / 2.0
        dist = tof_dc_a * SPEED_OF_LIGHT
        #print(a, b, dist)
        rssi = response_msg['rssi']

        yield adjust_dist_by_rssi(dist, rssi)

def calc_tdoa_alt_twr(l, a, b, history):

    assert l != a and l != b and a != b

    for seq in extract_message_sequences(l, history, [a, b, a, b, l, l, l], [[], [0], [1], [2], [0], [1], [2]]):
        (poll_msg, response_msg, final_msg, data_msg, poll_rx_msg, response_rx_msg, final_rx_msg) = seq

        # TODO: Check me again!
        poll_tx_ts_a = convert_ts_to_sec(poll_msg['tx']['ts'])
        poll_rx_ts_b = convert_ts_to_sec(msg_get_rx_ts(response_msg, poll_msg))
        poll_rx_ts_l = convert_ts_to_sec(msg_get_rx_ts(poll_rx_msg, poll_msg))

        response_tx_ts_b = convert_ts_to_sec(response_msg['tx']['ts'])
        response_rx_ts_a = convert_ts_to_sec(msg_get_rx_ts(final_msg, response_msg))
        response_rx_ts_l = convert_ts_to_sec(msg_get_rx_ts(response_rx_msg, response_msg))

        final_tx_ts_a = convert_ts_to_sec(final_msg['tx']['ts'])
        final_rx_ts_b = convert_ts_to_sec(msg_get_rx_ts(data_msg, final_msg))
        final_rx_ts_l = convert_ts_to_sec(msg_get_rx_ts(final_rx_msg, final_msg))

        assert None not in [poll_tx_ts_a, poll_rx_ts_b, poll_rx_ts_l, response_tx_ts_b, response_rx_ts_a, response_rx_ts_l, final_tx_ts_a, final_rx_ts_b, final_rx_ts_l]

        [poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a] = handle_overflow_in_sec([poll_tx_ts_a, response_rx_ts_a, final_tx_ts_a])
        [poll_rx_ts_b, response_tx_ts_b, final_rx_ts_b] = handle_overflow_in_sec([poll_rx_ts_b, response_tx_ts_b, final_rx_ts_b])
        [poll_rx_ts_l, response_rx_ts_l, final_rx_ts_l] = handle_overflow_in_sec([poll_rx_ts_l, response_rx_ts_l, final_rx_ts_l])

        round_a = response_rx_ts_a - poll_tx_ts_a
        delay_a = final_tx_ts_a - response_rx_ts_a

        #print("Delay A")
        #print(delay_a)
        #print(convert_ts_to_sec(118065201730- 68855340289))

        round_b = final_rx_ts_b - response_tx_ts_b
        delay_b = response_tx_ts_b - poll_rx_ts_b

        round_l = final_rx_ts_l - poll_rx_ts_l
        m_l = response_rx_ts_l - poll_rx_ts_l

        clock_ratio_ka_kb = (round_a + delay_a) / (delay_b + round_b)
        clock_ratio_kt_ka = round_l / (round_a + delay_a)

        tof_dc_a = (round_a-clock_ratio_ka_kb*delay_b) / 2.0
        tdoa = clock_ratio_kt_ka*(round_a-tof_dc_a)-m_l
        #print(l, a, b, tof_dc_a *SPEED_OF_LIGHT)
        yield tdoa*SPEED_OF_LIGHT
    return True

def parse_messages_from_lines(line_it):
    for line in line_it:
        if line.strip() == "":
            continue
        try:
            dev, json_str = line.split('\t', 2)

            if dev.find('/dev/tty') > 0:
                continue # some bogus thing happening here?
            try:
                msg = json.loads(json_str)
                yield (dev, msg)
            except json.decoder.JSONDecodeError:
                #print(json_str)
                pass
        except ValueError:
            pass


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

if __name__ == "__main__":
    dev_msg_iter = parse_messages_from_lines(sys.stdin)
    #dev_msg_iter = time_scaler(dev_msg_iter)

    msg_by_dev = {}
    dev_mapping = {}

    for (d, msg) in dev_msg_iter:
        if d not in msg_by_dev:
            msg_by_dev[d] = []

        if msg['type'] == 'tx':
            # We check that we did not generate the same device twice
            assert d not in dev_mapping or dev_mapping[d] == msg['tx']['addr']
            dev_mapping[d] = msg['tx']['addr']
        msg_by_dev[d].append(msg)

    tag_x = 0.90
    dev_positions = {
      # Left Anchors
      '/dev/tty.usbmodem0007601202631': (0, -0.05),
      '/dev/tty.usbmodem0007601202991': (0, 0),
      '/dev/tty.usbmodem0007601202711': (0, 0.05),

      # Right Anchors
      '/dev/tty.usbmodem0007601203521': (1.8, 0.05),
      '/dev/tty.usbmodem0007601203691': (1.8, 0),
      '/dev/tty.usbmodem0007601203181': (1.8, -0.05),

      # Tags
      '/dev/tty.usbmodem0007601203281': (tag_x, 0),
      '/dev/tty.usbmodem0007601202561': (tag_x, 0),
    }

    left_anchors = ['/dev/tty.usbmodem0007601202631', '/dev/tty.usbmodem0007601202991', '/dev/tty.usbmodem0007601202711']
    right_anchors = ['/dev/tty.usbmodem0007601203521', '/dev/tty.usbmodem0007601203691', '/dev/tty.usbmodem0007601203181']
    tags = ['/dev/tty.usbmodem0007601203281', '/dev/tty.usbmodem0007601202561']

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