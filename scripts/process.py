from typing import List
from serial import Serial
import threading
import time
import sys
import json
import itertools
import sys


OVERFLOW_OFFSET = 0xFFFFFFFFFF

def filtered_next(iter, fn):
    for x in iter:
        if fn(x):
            return x
    return None

#def filter_rx_for_tx(iter, tx_msg):
#    def has_rx_ts(msg):
#        return msg['tx']['addr'] == tx_msg['tx']['addr'] and msg['tx']['sn'] == tx_msg['tx']['sn']
#    return filtered_next(iter,  lambda m : has_rx_ts(m))

def msg_get_rx_ts(tx_msg, rx_msg):
    for msg in tx_msg['rx']:
        if msg['addr'] == rx_msg['tx']['addr'] and msg['sn'] == rx_msg['tx']['sn']:
            return msg['ts']
    return None


def filter_rx_for_tx(iter, tx_msg):
    return filtered_next(iter, lambda m: msg_get_rx_ts(m, tx_msg) is not None)

def filter_tx_for_rx(iter, rx_msg, alt):
    if alt is not None and msg_get_rx_ts(alt, rx_msg) is not None:
        return alt
    return filtered_next(iter,  lambda m : msg_get_rx_ts(m, rx_msg) is not None)

def convert_ts_to_sec(ts):
    return (ts*15650.0)*1e-15

# TODO: Shall we calculate the TDoA value from other devices as well?
def calc_tdoa_alt_twr(l, a, b, history):
    # we filter out received messages of anchor devices
    rx_hist_a = [msg for msg in history if msg['type'] == 'rx' and msg['tx']['addr'] == a]
    rx_hist_b = [ msg for msg in history if msg['type'] == 'rx' and msg['tx']['addr'] == b]
    tx_hist = [msg for msg in history if msg['type'] == 'tx']


    # we create iterators since we need to ensure that we do not reuse msgs
    rx_hist_a = iter(rx_hist_a)
    rx_hist_b = iter(rx_hist_b)
    tx_hist = iter(tx_hist)

    # we expect that A initiated the ranging
    poll_msg = next(rx_hist_a, None)
    if poll_msg is None:
        print("A")
        return None # seems like we received no init msg from A

    # the response needs to include the init_msg
    response_msg = filter_rx_for_tx(rx_hist_b, poll_msg)
    if response_msg is None:
        print("B")
        #print("response_msg is None")
        return None # seems like we did not hear back from B

    # the final message needs to include the response_msg
    final_msg = filter_rx_for_tx(rx_hist_a, response_msg)
    if final_msg is None:
        print("C")
        return None  # seems like A did not answer B (as far as L knows)

    # B then needs to disseminate the latest rx of the final_msg
    data_msg = filter_rx_for_tx(rx_hist_b, final_msg)
    if data_msg is None:
        print("D")
        #print("data_msg is None")
        return None  # seems B did not send its reception (as far as L knows)

    poll_rx_msg = filter_tx_for_rx(tx_hist, poll_msg, None)
    response_rx_msg = filter_tx_for_rx(tx_hist, response_msg, poll_rx_msg)
    final_rx_msg = filter_tx_for_rx(tx_hist, final_msg, response_rx_msg)

    if None in [poll_rx_msg, response_rx_msg, final_rx_msg]:
        print("E")
        #print("we could not find all timestamps!")
        return None # we could not find all timestamps!

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

    if not (poll_tx_ts_a < response_rx_ts_a < final_tx_ts_a):
        print("A not in order")

        if not(poll_tx_ts_a < response_rx_ts_a ):
            response_rx_ts_a += convert_ts_to_sec(OVERFLOW_OFFSET)
            final_tx_ts_a += convert_ts_to_sec(OVERFLOW_OFFSET)

        if not (response_rx_ts_a < final_tx_ts_a):
            final_tx_ts_a += convert_ts_to_sec(OVERFLOW_OFFSET)
        #return None

    if not (poll_rx_ts_b < response_tx_ts_b < final_rx_ts_b):
        print("B not in order")

        if not(poll_rx_ts_b < response_tx_ts_b ):
            response_tx_ts_b += convert_ts_to_sec(OVERFLOW_OFFSET)
            final_rx_ts_b += convert_ts_to_sec(OVERFLOW_OFFSET)

        if not(response_tx_ts_b < final_rx_ts_b):
            final_rx_ts_b += convert_ts_to_sec(OVERFLOW_OFFSET)

        #return None

    if not (poll_rx_ts_l < response_rx_ts_l < final_rx_ts_l):
        print("L not in order")

        if not (poll_rx_ts_l < response_rx_ts_l):
            response_rx_ts_l += convert_ts_to_sec(OVERFLOW_OFFSET)
            final_rx_ts_l += convert_ts_to_sec(OVERFLOW_OFFSET)

        if not (response_rx_ts_l < final_rx_ts_l):
            final_rx_ts_l += convert_ts_to_sec(OVERFLOW_OFFSET)

        #return None

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
    #print(round_a)
    #print(clock_ratio_ka_kb*delay_b)

    clock_ratio_kt_ka = round_l / (round_a + delay_a)
    tof_dc_a = (round_a-clock_ratio_ka_kb*delay_b) / 2.0
    print(a,b, tof_dc_a * 299792458.0)

    # TODO: Filter the rx list of the listener for the CFO!

    return True


def calc_tdoa_ss_twr(l, a, b, rx_history):
    return None

def parse_messages_from_lines(line_it):
    for line in line_it:
        if line.strip() == "":
            continue
        try:
            dev, json_str = line.split('\t', 2)
            msg = json.loads(json_str)
            yield (dev, msg)
        except json.decoder.JSONDecodeError:
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


if __name__ == "__main__":

    dev_msg_iter = parse_messages_from_lines(sys.stdin)
    #dev_msg_iter = time_scaler(dev_msg_iter)

    msg_by_dev = {}
    dev_mapping = {}

    for (d, msg) in dev_msg_iter:
        if d not in msg_by_dev:
            msg_by_dev[d] = []

        if msg['type'] == 'tx':
            dev_mapping[d] = msg['tx']['addr']
        msg_by_dev[d].append(msg)


        # if we have a tx message, we try to compute the TDoA values with everyone else
        if msg['type'] == 'tx' and d == '/dev/tty.usbmodem0007601202991':
            for (a, b) in itertools.combinations(msg_by_dev, 2):
                if d in [a, b]:
                    continue
                if a not in dev_mapping or b not in dev_mapping:
                    continue
                calc_tdoa_alt_twr(dev_mapping[d], dev_mapping[a], dev_mapping[b], msg_by_dev[d][-8:])
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