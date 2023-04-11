import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base import pair_index

NODES_POSITIONS = [
    (0,0),
    (11,0),
    (22,0),
    (22,7.5),
    (22,15),
    (11,15),
    (0,15),
    (0,7.5)
]

PASSIVE_NODE_POSITION=(5,5)

# Speed of light in air
c_in_air = 299702547.236
RESP_DELAY_S = 0.005

NODE_DRIFT_STD = 10.0/1000000.0

TX_DELAY_MEAN = 0.516e-09
TX_DELAY_STD = 0.06e-09
RX_DELAY_MEAN = TX_DELAY_MEAN
RX_DELAY_STD = TX_DELAY_STD
RX_NOISE_STD = 1.0e-09

# Enable perfect measurements but with drift!
# TX_DELAY_MEAN = 0
# TX_DELAY_STD = 0.01e-09
# RX_DELAY_MEAN = 0
# RX_DELAY_STD = 0.01e-09
# RX_NOISE_STD = 0
# NODE_DRIFT_STD = 0.0


import time


def dist(a, b):
    return np.linalg.norm(np.array(NODES_POSITIONS[a]) - np.array(NODES_POSITIONS[b]))

def tof(a, b):
    return dist(a, b) / c_in_air


def tof_to_passive(a):
    return np.linalg.norm(np.array(NODES_POSITIONS[a]) - np.array(PASSIVE_NODE_POSITION)) / c_in_air


def create_inference_matrix(n):
    X = np.zeros((int(n * (n - 1) / 2), n))

    row = 0
    for a in range(n):
        for b in range(n):
            if b < a:
                X[row][a] = 1
                X[row][b] = 1
                row += 1

    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X))
    return B

INF_MAT = create_inference_matrix(len(NODES_POSITIONS))


def create_inference_matrix_rep(n, rep=1):
    X = np.zeros((int(n * (n - 1) / 2)*rep, n))

    row = 0
    for a in range(n):
        for b in range(n):
            if b < a:
                for i in range(rep):
                    X[row][a] = 1
                    X[row][b] = 1
                    row += 1

    B = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X))
    return B



def sim_exchange(num_total_exchanges):
    n = len(NODES_POSITIONS)
    # we fix the node drifts
    node_drifts = np.random.normal(loc=1.0,  scale=NODE_DRIFT_STD, size=n)
    passive_node_drift = np.random.normal(loc=1.0,  scale=NODE_DRIFT_STD, size=1)

    tx_delays = np.random.normal(loc=TX_DELAY_MEAN,  scale=TX_DELAY_STD, size=n)
    rx_delays = np.random.normal(loc=RX_DELAY_MEAN,  scale=RX_DELAY_STD, size=n)

    rounds = []
    for i in range(num_total_exchanges):
        round = []

        for a in range(len(NODES_POSITIONS)):
            for b in range(len(NODES_POSITIONS)):
                if b < a:
                    # a initates the ranging
                    t = tof(a, b)

                    a_actual_poll_tx = 0
                    b_actual_poll_rx = a_actual_poll_tx + np.random.normal(loc=t, scale=RX_NOISE_STD)
                    b_actual_response_tx = b_actual_poll_rx + RESP_DELAY_S
                    a_actual_response_rx = b_actual_response_tx + np.random.normal(loc=t, scale=RX_NOISE_STD)
                    a_actual_final_tx = a_actual_response_rx + RESP_DELAY_S
                    b_actual_final_rx = a_actual_final_tx + np.random.normal(loc=t, scale=RX_NOISE_STD)

                    # tx timestamps are skewed in a negative way -> i.e. increase the measured_rtt
                    a_delayed_poll_tx = a_actual_poll_tx - tx_delays[a]
                    b_delayed_response_tx = b_actual_response_tx - tx_delays[b]
                    a_delayed_final_tx = a_actual_final_tx - tx_delays[a]

                    # rx timestamps are skewed in a positive way
                    b_delayed_poll_rx = b_actual_poll_rx + rx_delays[b]
                    a_delayed_response_rx = a_actual_response_rx + rx_delays[a]
                    b_delayed_final_rx = b_actual_final_rx + rx_delays[b]

                    a_measured_round_undrifted = a_delayed_response_rx - a_delayed_poll_tx
                    b_measured_round_undrifted = b_delayed_final_rx - b_delayed_response_tx
                    a_measured_delay_undrifted = a_delayed_final_tx - a_delayed_response_rx
                    b_measured_delay_undrifted = b_delayed_response_tx - b_delayed_poll_rx

                    # we compute times for TDoA using an additional passive node, note that we do not need delays here
                    p_actual_poll_rx = a_actual_poll_tx + np.random.normal(loc=tof_to_passive(a), scale=RX_NOISE_STD)
                    p_actual_response_rx = b_actual_response_tx + np.random.normal(loc=tof_to_passive(b), scale=RX_NOISE_STD)
                    p_actual_final_rx = a_actual_final_tx + np.random.normal(loc=tof_to_passive(a), scale=RX_NOISE_STD)

                    round.append({
                        "device_a": a,
                        "device_b": b,
                        "drift_a":  node_drifts[a],
                        "drift_b":  node_drifts[b],
                        "drift_p":  passive_node_drift,
                        "round_a": a_measured_round_undrifted * node_drifts[a],
                        "delay_a": a_measured_delay_undrifted * node_drifts[a],
                        "round_b": b_measured_round_undrifted * node_drifts[b],
                        "delay_b": b_measured_delay_undrifted * node_drifts[b],
                        "passive_tdoa": (p_actual_response_rx-p_actual_poll_rx) * passive_node_drift,
                        "passive_overall": (p_actual_final_rx-p_actual_poll_rx) * passive_node_drift,
                    })

        rounds.append(round)
    return (node_drifts, tx_delays, rx_delays, rounds)


def calc_simple(ex, comb_delay=0.0):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    relative_drift = (round_a+delay_a)/(round_b+delay_b)
    return (round_a-delay_b*relative_drift-comb_delay) * 0.5






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



def calibrate_delays_gn(rounds, num_iterations=100):
    actual = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                actual.append(tof(a, b))
    actual = np.array(actual)

    est_combined_delays = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                pi = pair_index(a,b)
                est_combined_delay = RX_DELAY_MEAN+TX_DELAY_MEAN
                exchanges = []
                for r in rounds:
                    exchanges.append(r[pi])

                def compute_h(delay):
                    xs = []
                    for e in exchanges:
                        xs.append(calc_complex_tof_d_comb(e, delay))
                    return np.array(xs).transpose()

                def compute_T(delay):
                    xs = []
                    for e in exchanges:
                        xs.append(calc_complex_tof(e, delay))
                    return actual[pi] - np.array(xs)


                for i in range(num_iterations):
                    h = compute_h(est_combined_delay)
                    T = compute_T(est_combined_delay)

                    # we normalize h
                    h_normed = h / np.linalg.norm(h)

                    #old_est_combined_delay = est_combined_delay
                    est_combined_delay = est_combined_delay + np.matmul(h_normed, T) * 0.1 # we adjust the learning rate
                    #est_combined_delay_new = est_combined_delay - np.mean(T)
                    #print(len(exchanges), old_est_combined_delay, est_combined_delay)
                    #print(old_est_combined_delay, np.matmul(h_normed, T), est_combined_delay)

                    if abs(est_combined_delay) > RX_NOISE_STD*10.0:
                        print("GN might not converge!", est_combined_delay)
                        # print(a,b,i)
                        # print(h)
                        # print(h_normed)
                        # print(T)
                        # print(T.dtype, h.dtype)
                        # print("matmul_normed", np.matmul(h_normed, T))
                        # print("matmul", np.matmul(h, T))
                        # print("mean", np.sum(T)*h_normed[0])

                        #print(old_est_combined_delay, est_combined_delay, est_combined_delay_new)
                est_combined_delays.append(est_combined_delay)
    est_combined_delays = np.array(est_combined_delays)

    delays = np.matmul(INF_MAT, np.transpose(est_combined_delays))

    return np.transpose(delays)




def calibrate_delays_our_approach(rounds, device_drifts, source_device=0):

    sums = []
    actual = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                sums.append(0)
                actual.append(tof(a,b))

    individual_measurements = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                pi = pair_index(a,b)
                for r in rounds:
                    ex = r[pi]
                    tof_calc_in_time_a = calc_simple(ex)
                    tof_calc_in_source_time = device_drifts[source_device] * tof_calc_in_time_a / device_drifts[a]
                    sums[pi] += tof_calc_in_source_time
                    individual_measurements.append(tof_calc_in_source_time)

    sums = np.array(sums)
    actual = np.array(actual)

    sums /= len(rounds)

    diffs = sums - actual
    diffs *= 2.0

    delays = np.matmul(INF_MAT, np.transpose(diffs))

    return np.transpose(delays)  # list of antenna delays


def calibrate_delays_our_approach_via_source_device(rounds, source_device=0):
    sums = []
    actual = []
    assert (source_device==0)
    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                sums.append(0)
                actual.append(tof(a, b))

    individual_diffs = []
    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                pi = pair_index(a, b)
                for r in rounds:
                    ex_a_b = r[pi]

                    # we calculate the simple one from a to b
                    t = calc_simple(ex_a_b, comb_delay=0.0)

                    if source_device != a:
                        ex_s_a = r[pair_index(source_device, a)]
                        # then we convert it to the source time
                        # TODO: we only support source device 0 for now (because of the ranging sides...)
                        (round_s, delay_s, round_a, delay_a) = (ex_s_a['round_a'], ex_s_a['delay_a'], ex_s_a['round_b'], ex_s_a['delay_b'])
                        t = (t * (round_s + delay_s)) / (round_a + delay_a)

                    sums[pi] += t
                    #individual_diffs.append((t-tof(a, b))*2.0)

    sums = np.array(sums)
    actual = np.array(actual)

    sums /= len(rounds)

    diffs = sums - actual
    diffs *= 2.0

    delays = np.matmul(INF_MAT, np.transpose(diffs))

    # print("TEST", len(rounds))
    # print(delays)
    #
    # individual_diffs = np.array(individual_diffs)
    # delays = np.matmul(create_inference_matrix_rep(len(NODES_POSITIONS), len(rounds)),
    #                    np.transpose(individual_diffs))
    # print(delays)

    return np.transpose(delays)  # list of antenna delays



def calibrate_delays_tdoa(rounds):
    # we use the last device since it initiates all rangings
    # we will calculate the master node independently
    m = len(NODES_POSITIONS) - 1


    delays = []

    # Calibrate devices independently
    for a in range(len(NODES_POSITIONS)-1):
        pi = pair_index(m, a)

        delays_a = []
        delays_m = []

        for r in rounds:
            ex_m_a = r[pi]

            assert ex_m_a['device_a'] == m
            assert ex_m_a['device_b'] == a

            (round_m, delay_m, round_a, delay_a) = (ex_m_a['round_a'], ex_m_a['delay_a'], ex_m_a['round_b'], ex_m_a['delay_b'])
            (passive_tdoa, passive_overall) = (ex_m_a['passive_tdoa'], ex_m_a['passive_overall'])


            #tof(m, a), tof_to_passive(m), tof_to_passive(a)

            est_delay_m = round_m - (passive_tdoa * ((round_m+delay_m)/passive_overall)) + tof_to_passive(a) - tof(m, a) - tof_to_passive(m)
            est_delay_a = (passive_tdoa * (round_m+delay_m)/passive_overall) - delay_a * ((round_m+delay_m)/(round_a+delay_a)) - tof_to_passive(a) - tof(m, a) + tof_to_passive(m)

            delays_a.append(est_delay_a)
            delays_m.append(est_delay_m)

        delays.append(np.mean(delays_a))

        # we add the master delay in the end
        if a == m - 1:
            delays.append(np.mean(delays_m))
    return np.transpose(delays)  # list of antenna delays

# we get a list of dictionaries, containing the measurement pairs

xs = [16, 64, 256, 1024]




    # (node_drifts, tx_delays, rx_delays, rounds) = exchanges
    #
    # compl_errors = []
    # simple_errors = []
    # for r in rounds:
    #     i = 0
    #     for a in range(len(NODES_POSITIONS)):
    #         for b in range(len(NODES_POSITIONS)):
    #             if b < a:
    #                 actual_dist = tof(a,b)
    #                 compl = calc_complex_tof(r[i])
    #                 simple = calc_simple(r[i])
    #                 compl_errors.append(compl-actual_dist)
    #                 simple_errors.append(simple-actual_dist)
    #                 i += 1
    #
    # compl_errors = np.array(compl_errors)
    # simple_errors = np.array(simple_errors)
    #
    # compl_error_rmse = np.sqrt(((compl_errors) ** 2).mean())
    # simple_error_rmse = np.sqrt(((simple_errors) ** 2).mean())
    # print(compl_errors.mean(), simple_errors.mean())
    # print(compl_error_rmse, simple_error_rmse)
    # exit()


def get_sim_data_rows(xs=[16, 64, 256, 1024], num_repetitiosn=1):

    max_exchanges = max(xs)
    repetitions = []
    for i in range(num_repetitiosn):
        exchanges = sim_exchange(num_total_exchanges=max_exchanges)
        repetitions.append(exchanges)

    data_rows = []

    for x in xs:

        rmses_tdoa = []
        rmses_gn = []
        rmses_our = []
        tdoa_times = []
        gn_times = []
        our_times = []

        for i in range(num_repetitiosn):

            (node_drifts, tx_delays, rx_delays, rounds) = repetitions[i]

            rounds = rounds[0:x]

            actual_delays = tx_delays + rx_delays

            ts = time.time()
            gn_est_delays = calibrate_delays_gn(rounds=rounds)
            te = time.time()
            gn_times.append(te-ts)

            ts = time.time()
            tdoa_est_delays = calibrate_delays_tdoa(rounds=rounds)
            te = time.time()
            tdoa_times.append(te - ts)

            ts = time.time()
            our_est_delays = calibrate_delays_our_approach_via_source_device(rounds=rounds, source_device = 0)
            te = time.time()
            our_times.append(te - ts)

            tdoa_err = np.sqrt(((tdoa_est_delays-actual_delays)** 2).mean())
            gn_err = np.sqrt(((gn_est_delays-actual_delays)** 2).mean())
            our_err = np.sqrt(((our_est_delays-actual_delays)** 2).mean())

            rmses_tdoa.append(tdoa_err)
            rmses_gn.append(gn_err)
            rmses_our.append(our_err)

        rmses_tdoa = np.array(rmses_tdoa)
        rmses_gn = np.array(rmses_gn)
        rmses_our = np.array(rmses_our)

        gn_times = np.array(gn_times)
        our_times = np.array(our_times)


        data_rows.append({
            'num_measurements': x,
            'tdoa_mean': rmses_tdoa.mean() * c_in_air * 100,
            'tdoa_std': rmses_tdoa.std() * c_in_air * 100,
            'gn_mean': rmses_gn.mean()* c_in_air*100,
            'gn_std': rmses_gn.std()* c_in_air*100,
            'our_mean': rmses_our.mean() * c_in_air * 100,
            'our_std': rmses_our.std() * c_in_air * 100,
            'gn_time_mean': gn_times.mean(),
            'our_time_mean': our_times.mean(),
            'speedup': gn_times.mean() / our_times.mean(),
            'speedup_err': 0,
        })
    return data_rows



# print(x, rmses_gn.mean() * c_in_air*100, rmses_our.mean()* c_in_air*100, rmses_gn.std() * c_in_air*100, rmses_our.std() * c_in_air*100)
# #print(rmses_tdoa)
# #print(rmses_gn)
#
#
#
# #print(exchanges)
#
# first = rounds[0][0]
#
# actual = dist(1, 0)
# simple = calc_simple(first)*c_in_air
# com = calc_complex_tof(first)*c_in_air
#
# #print(actual, simple, com)
#
# actual = dist(1, 0)
# simple = calc_simple(first, comb_delay=rx_delays[0]+rx_delays[1]+tx_delays[0]+tx_delays[1]) * c_in_air
# com = calc_complex_tof(first, comb_delay=rx_delays[0]+rx_delays[1]+tx_delays[0]+tx_delays[1]) * c_in_air
# #print(actual, simple, com)
