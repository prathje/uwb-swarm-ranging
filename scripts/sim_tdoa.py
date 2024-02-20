import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base import pair_index

NODES_POSITIONS = [
    (0,0),
    (10,0),
    #(22,0),
    #(22,7.5),
    #(22,15),
    #(11,15),
    #(0,15),
    #(0,7.5)
]

PASSIVE_NODE_POSITION=(7.5,5)

# Speed of light in air
c_in_air = 299702547.236
RESP_DELAY_S = 0.05

NODE_DRIFT_STD = 10.0/1000000.0

TX_DELAY_MEAN = 0.0
TX_DELAY_STD = 0.0
RX_DELAY_MEAN = 0.0
RX_DELAY_STD = 0.0

RX_NOISE_STD = 1.0e-09

DRIFT_RATE_STD = 0.0

# Enable perfect measurements but with drift!
# TX_DELAY_MEAN = 0
# TX_DELAY_STD = 0.01e-09
# RX_DELAY_MEAN = 0
# RX_DELAY_STD = 0.01e-09
# RX_NOISE_STD = 0
# node_drift_std = 0.0


import time

def dist(a, b):
    return np.linalg.norm(np.array(NODES_POSITIONS[a]) - np.array(NODES_POSITIONS[b]))

def tof(a, b):
    return dist(a, b) / c_in_air


def tof_to_passive(a):
    return np.linalg.norm(np.array(NODES_POSITIONS[a]) - np.array(PASSIVE_NODE_POSITION)) / c_in_air


def sim_exchange(a, b, resp_delay_s=RESP_DELAY_S, node_drift_std=NODE_DRIFT_STD, tx_delay_mean=TX_DELAY_MEAN, tx_delay_std=TX_DELAY_STD, rx_delay_mean=RX_DELAY_MEAN, rx_delay_std=RX_DELAY_STD, rx_noise_std=RX_NOISE_STD, drift_rate_std=DRIFT_RATE_STD):
    n = len(NODES_POSITIONS)
    # we fix the node drifts
    node_drifts = np.random.normal(loc=1.0,  scale=node_drift_std, size=n)
    passive_node_drift = np.random.normal(loc=1.0,  scale=node_drift_std)

    tx_delays = np.random.normal(loc=tx_delay_mean,  scale=tx_delay_std, size=n)
    rx_delays = np.random.normal(loc=rx_delay_mean,  scale=rx_delay_std, size=n)

    # a initates the ranging
    t = tof(a, b)

    if isinstance(resp_delay_s, tuple):
        resp_delay_s_a, resp_delay_s_b = resp_delay_s
    else:
        resp_delay_s_a = resp_delay_s
        resp_delay_s_b = resp_delay_s

    def get_rx_noise(tx, rx):
        if isinstance(rx_noise_std, dict):
            return rx_noise_std["{}-{}".format(tx, rx)]
        else:
            return rx_noise_std

    def calc_drifted_dur_a(dur):
        return dur * node_drifts[a] + np.random.normal(loc=0.0, scale=drift_rate_std*dur)

    def calc_drifted_dur_b(dur):
        return dur * node_drifts[b] + np.random.normal(loc=0.0, scale=drift_rate_std*dur)

    def calc_drifted_dur_passive(dur):
        return dur * passive_node_drift + np.random.normal(loc=0.0, scale=drift_rate_std*dur)

    a_actual_poll_tx = 0
    b_actual_poll_rx = a_actual_poll_tx + np.random.normal(loc=t, scale=get_rx_noise('a', 'b'))
    b_actual_response_tx = b_actual_poll_rx + resp_delay_s_a
    a_actual_response_rx = b_actual_response_tx + np.random.normal(loc=t, scale=get_rx_noise('b', 'a'))
    a_actual_final_tx = a_actual_response_rx + resp_delay_s_b
    b_actual_final_rx = a_actual_final_tx + np.random.normal(loc=t, scale=get_rx_noise('a', 'b'))


    # tx timestamps are skewed in a negative way -> i.e. increase the measured_rtt
    a_delayed_poll_tx = a_actual_poll_tx - tx_delays[a]
    b_delayed_response_tx = b_actual_response_tx - tx_delays[b]
    a_delayed_final_tx = a_actual_final_tx - tx_delays[a]

    # rx timestamps are skewed in a positive way
    b_delayed_poll_rx = b_actual_poll_rx + rx_delays[b]
    a_delayed_response_rx = a_actual_response_rx + rx_delays[a]
    b_delayed_final_rx = b_actual_final_rx + rx_delays[b]

    a_measured_round_drifted = calc_drifted_dur_a(a_delayed_response_rx - a_delayed_poll_tx)
    b_measured_round_drifted = calc_drifted_dur_b(b_delayed_final_rx - b_delayed_response_tx)
    a_measured_delay_drifted = calc_drifted_dur_a(a_delayed_final_tx - a_delayed_response_rx)
    b_measured_delay_drifted = calc_drifted_dur_b(b_delayed_response_tx - b_delayed_poll_rx)

    # we compute times for TDoA using an additional passive node, note that we do not need delays here
    p_actual_poll_rx = a_actual_poll_tx + np.random.normal(loc=tof_to_passive(a), scale=get_rx_noise('a', 'p'))
    p_actual_response_rx = b_actual_response_tx + np.random.normal(loc=tof_to_passive(b), scale=get_rx_noise('b', 'p'))
    p_actual_final_rx = a_actual_final_tx + np.random.normal(loc=tof_to_passive(a), scale=get_rx_noise('a', 'p'))

    passive_tdoa_drifted = calc_drifted_dur_passive(p_actual_response_rx - p_actual_poll_rx)
    passive_overall_drifted = passive_tdoa_drifted+calc_drifted_dur_passive(p_actual_final_rx - p_actual_response_rx)


    return {
        "device_a": a,
        "node_drifts": node_drifts,
        "passive_node_drift": passive_node_drift,
        "device_b": b,
        "drift_a": node_drifts[a],
        "drift_b": node_drifts[b],
        "drift_p": passive_node_drift,
        "round_a": a_measured_round_drifted,
        "delay_a": a_measured_delay_drifted,
        "round_b": b_measured_round_drifted,
        "delay_b": b_measured_delay_drifted,
        "passive_tdoa": passive_tdoa_drifted,
        "passive_overall": passive_overall_drifted,
    }


def calc_simple(ex, comb_delay=0.0, mitigate_drift=True):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    relative_drift = (round_a+delay_a)/(round_b+delay_b)

    if not mitigate_drift:
        relative_drift = 1.0

    return (round_a-delay_b*relative_drift-comb_delay) * 0.5


def calc_simple_ds(ex, mitigate_drift=True):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    relative_drift = (round_a+delay_a)/(round_b+delay_b)

    if not mitigate_drift:
        relative_drift = 1.0

    return (round_a-delay_b*relative_drift) * 0.25 + (round_b*relative_drift-delay_a) * 0.25

def calc_tdoa_simple(ex, mitigate_drift=True):

    rel_cd_a = ex['passive_overall'] / (ex['round_a'] + ex['delay_a'])
    rel_cd_b = rel_cd_a * ((ex['round_a'] + ex['delay_a']) / (ex['round_b'] + ex['delay_b'])) # we reuse the cd between passive and a to estimate the relative drift to b

    if not mitigate_drift:
         rel_cd_a = 1.0
         rel_cd_b = 1.0

    tdoa = 0.5*rel_cd_a*ex['round_a'] + 0.5*rel_cd_b*ex['delay_b'] - ex['passive_tdoa']

    return tdoa

def calc_tdoa_simple_2(ex, mitigate_drift=True):

    rel_cd_a = ex['passive_overall'] / (ex['round_a'] + ex['delay_a'])
    rel_cd_b = ex['passive_overall'] / (ex['round_b'] + ex['delay_b'])

    if not mitigate_drift:
         rel_cd_a = 1.0
         rel_cd_b = 1.0

    tdoa = 0.5*rel_cd_a*ex['round_a'] + 0.5*rel_cd_b*ex['delay_b'] - ex['passive_tdoa']

    return tdoa

def calc_tdoa_ds(ex, mitigate_drift=True):

    rel_cd_a = ex['passive_overall'] / (ex['round_a'] + ex['delay_a'])
    relative_drift = ((ex['round_a'] + ex['delay_a']) / (ex['round_b'] + ex['delay_b']))
    rel_cd_b = rel_cd_a * relative_drift # we reuse the cd between passive and a to estimate the relative drift to b

    if not mitigate_drift:
         rel_cd_a = 1.0
         rel_cd_b = 1.0

    tdoa_one = 0.5*rel_cd_a*ex['round_a'] + 0.5*rel_cd_b*ex['delay_b'] - ex['passive_tdoa']
    tdoa_two = 0.5*rel_cd_a*ex['delay_a'] + 0.5*rel_cd_b*ex['round_b'] - (ex['passive_overall']-ex['passive_tdoa'])

    return (tdoa_one - tdoa_two) * 0.5


def calc_tdoa_tof_half_corrected(ex, mitigate_drift=True):

    # we only correct for the tof relative drift here!
    relative_drift = ((ex['round_a'] + ex['delay_a']) / (ex['round_b'] + ex['delay_b']))

    if not mitigate_drift:
         relative_drift = 1.0

    tdoa_one = 0.5*ex['round_a'] + 0.5*relative_drift*ex['delay_b'] - ex['passive_tdoa']

    return tdoa_one

def calc_tdoa_ds_tof_half_corrected(ex, mitigate_drift=True):

    # we only correct for the tof relative drift here!
    relative_drift = ((ex['round_a'] + ex['delay_a']) / (ex['round_b'] + ex['delay_b']))

    if not mitigate_drift:
         relative_drift = 1.0

    tdoa_one = 0.5*ex['round_a'] + 0.5*relative_drift*ex['delay_b'] - ex['passive_tdoa']
    tdoa_two = 0.5*ex['delay_a'] + 0.5*relative_drift*ex['round_b'] - (ex['passive_overall']-ex['passive_tdoa'])

    return (tdoa_one-tdoa_two)*0.5


#def calc_simple_2(ex, comb_delay=0.0):
#    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
#    #return ((round_b + delay_b)*round_a - delay_b * (round_a + delay_a) - (round_b + delay_b)*comb_delay) / (2.0*(round_b + delay_b))
#    return ((round_b * round_a - delay_b * delay_a) - (round_b + delay_b)*comb_delay) / (2.0*(round_b + delay_b))


def calc_complex_tof(ex, comb_delay=0.0):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    (a, x, b, y) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])

    #print ((a*b-x*y-(x+y)*c-c*c) / (x+y+a+b+2*c)-(round_a*round_b-delay_a*delay_b-(delay_a+delay_b)*comb_delay-comb_delay*comb_delay) / (delay_a+delay_b+round_a+round_b+2*comb_delay))
    return (round_a*round_b-delay_a*delay_b-(delay_a+delay_b)*comb_delay-comb_delay*comb_delay) / (delay_a+delay_b+round_a+round_b+2*comb_delay)

def calc_complex_tof_d_comb(ex, comb_delay=0.0):
    (a, x, b, y) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    c = comb_delay
    #d/dc((a b - x y - (x + y) c - c c)/(x + y + a + b + 2 c)) =
    return (-(2*c + x + y)*(a + b + 2*c + x + y) - 2*a*b + 2*pow(c,2) + 2*c*(x + y) + 2*x*y) / pow(a + b + 2*c + x + y, 2)


def calculate_in_place(data_rows, mitigate_drift=True):
    for r in data_rows:

        # calculate tof in terms of a's clock using alt-ds calculations
        r['est_tof_a'] = calc_simple(r, mitigate_drift=mitigate_drift)
        r['est_tof_a_ds'] = calc_simple_ds(r, mitigate_drift=mitigate_drift)
        r['est_tdoa'] = calc_tdoa_simple_2(r, mitigate_drift=mitigate_drift)
        r['est_tdoa_half_cor'] = calc_tdoa_tof_half_corrected(r, mitigate_drift=mitigate_drift)
        r['est_tdoa_ds'] = calc_tdoa_ds(r, mitigate_drift=mitigate_drift)
        r['est_tdoa_ds_half_cor'] = calc_tdoa_ds_tof_half_corrected(r, mitigate_drift=mitigate_drift)

        r['real_tof'] = dist(r['device_a'], r['device_b']) / c_in_air
        r['real_tdoa'] = tof_to_passive(r['device_a']) - tof_to_passive(r['device_b'])

        yield r


def sim(num_exchanges = 100000, resp_delay_s=RESP_DELAY_S, node_drift_std=NODE_DRIFT_STD, tx_delay_mean=TX_DELAY_MEAN, tx_delay_std=TX_DELAY_STD, rx_delay_mean=RX_DELAY_MEAN, rx_delay_std=RX_DELAY_STD, rx_noise_std=RX_NOISE_STD, mitigate_drift=True, drift_rate_std=DRIFT_RATE_STD):

    data_rows = list(calculate_in_place([sim_exchange(0, 1, resp_delay_s=resp_delay_s, node_drift_std = node_drift_std, tx_delay_mean = tx_delay_mean, tx_delay_std = tx_delay_std, rx_delay_mean = rx_delay_mean, rx_delay_std = rx_delay_std, rx_noise_std = rx_noise_std, drift_rate_std=drift_rate_std) for x in range(num_exchanges)], mitigate_drift= mitigate_drift))

    data = {}
    for k in data_rows[0]:
        data[k] = np.asarray([r[k] for r in data_rows])

    return data, data_rows

