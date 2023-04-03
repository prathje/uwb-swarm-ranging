import numpy as np

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

# Speed of light in air, NOT USED ATM
c_in_air = 299702547.236
RESP_DELAY_S = 0.005


NUM_TOTAL_EXCHANGES = 2048
NUM_REPETITIONS = 1


def dist(a, b):
    return np.linalg.norm(np.array(NODES_POSITIONS[a]) - np.array(NODES_POSITIONS[b]))

def tof(a, b):
    return dist(a, b) / c_in_air


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



def sim_exchange():
    n = len(NODES_POSITIONS)
    # we fix the node drifts
    node_drifts = np.random.normal(loc=1.0,  scale=10.0/1000000.0, size=n)

    tx_delays = np.random.normal(loc=0.516e-09,  scale=0.06e-09, size=n)
    rx_delays = np.random.normal(loc=0.516e-09,  scale=0.06e-09, size=n)

    rounds = []
    for i in range(NUM_TOTAL_EXCHANGES):
        round = []

        for a in range(len(NODES_POSITIONS)):
            for b in range(len(NODES_POSITIONS)):
                if b < a:
                    # a initates the ranging
                    t = tof(a, b)

                    a_actual_poll_tx = 0
                    b_actual_poll_rx = a_actual_poll_tx + np.random.normal(loc=t, scale=1.0e-09)
                    b_actual_response_tx = b_actual_poll_rx + RESP_DELAY_S
                    a_actual_response_rx = b_actual_response_tx + np.random.normal(loc=t, scale=1.0e-09)
                    a_actual_final_tx = a_actual_response_rx + RESP_DELAY_S
                    b_actual_final_rx = a_actual_final_tx + np.random.normal(loc=t, scale=1.0e-09)

                    # tx timestamps are skewed in a negative way
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

                    round.append({
                        "round_a": a_measured_round_undrifted * node_drifts[a],
                        "delay_a": a_measured_delay_undrifted * node_drifts[a],
                        "round_b": b_measured_round_undrifted * node_drifts[b],
                        "delay_b": b_measured_delay_undrifted * node_drifts[b],
                    })
        rounds.append(round)
    return (node_drifts, tx_delays, rx_delays, rounds)


def calc_simple(ex, comb_delay=0.0):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    relative_drift = (round_a+delay_a)/(round_b+delay_b)
    return (round_a-delay_b*relative_drift-comb_delay) * 0.5


def calc_complex_tof(ex, comb_delay=0.0):
    (round_a, delay_a, round_b, delay_b) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    (a, x, b, y) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])

    c = comb_delay

    print ((a*b-x*y-(x+y)*c-c*c) / (x+y+a+b+2*c)-(round_a*round_b-delay_a*delay_b-(delay_a+delay_b)*comb_delay-comb_delay*comb_delay) / (delay_a+delay_b+round_a+round_b+2*comb_delay))
    return (round_a*round_b-delay_a*delay_b-(delay_a+delay_b)*comb_delay-comb_delay*comb_delay) / (delay_a+delay_b+round_a+round_b+2*comb_delay)

def calc_complex_tof_d_comb(ex, comb_delay=0.0):
    (a, x, b, y) = (ex['round_a'], ex['delay_a'], ex['round_b'], ex['delay_b'])
    c = comb_delay
    #d/dc((a b - x y - (x + y) c - c c)/(x + y + a + b + 2 c)) =
    return (-(2*c + x + y)*(a + b + 2*c + x + y) - 2*a*b + 2*pow(c,2) + 2*c*(x + y) + 2*x*y) / pow(a + b + 2*c + x + y, 2)


def calibrate_delays_pso(rounds):
    return np.array([0]*len(NODES_POSITIONS))   # list of antenna delays

def calibrate_delays_gn(rounds):
    return np.array([0]*len(NODES_POSITIONS))   # list of antenna delays





def calibrate_delays_our_approach(rounds):

    sums = []
    actual = []

    for a in range(len(NODES_POSITIONS)):
        for b in range(len(NODES_POSITIONS)):
            if b < a:
                sums.append(0)
                actual.append(tof(a,b))

    for r in rounds:
        i = 0
        for a in range(len(NODES_POSITIONS)):
            for b in range(len(NODES_POSITIONS)):
                if b < a:
                    ex = r[i]
                    sums[i] += calc_simple(ex)
                    i += 1

    sums = np.array(sums)
    actual = np.array(actual)

    sums /= len(rounds)

    diffs = sums - actual
    diffs *= 2.0

    delays = np.matmul(INF_MAT, np.transpose(diffs))

    return np.transpose(delays)  # list of antenna delays

# we get a list of dictionaries, containing the measurement pairs

xs = [4, 8,16,32,64,128,256,512,1024,2048]
ys_pso = []
ys_gn = []
ys_our = []

repetitions = []
for i in range(NUM_REPETITIONS):
    repetitions.append(sim_exchange())


for x in xs:

    rmses_pso = []
    rmses_gn = []
    rmses_our = []

    for i in range(NUM_REPETITIONS):

        (node_drifts, tx_delays, rx_delays, rounds) = repetitions[i]

        rounds = rounds[0:x]

        actual_delays = tx_delays + rx_delays

        pso_est_delays = calibrate_delays_pso(rounds=rounds)
        gn_est_delays = calibrate_delays_gn(rounds=rounds)
        our_est_delays = calibrate_delays_our_approach(rounds=rounds)

        pso_err = np.sqrt(((pso_est_delays-actual_delays)** 2).mean())
        gn_err = np.sqrt(((gn_est_delays-actual_delays)** 2).mean())
        our_err = np.sqrt(((our_est_delays-actual_delays)** 2).mean())

        rmses_pso.append(pso_err)
        rmses_gn.append(gn_err)
        rmses_our.append(our_err)

    rmses_pso = np.array(rmses_pso)
    rmses_gn = np.array(rmses_gn)
    rmses_our = np.array(rmses_our)



    print(x, rmses_our.mean()* c_in_air*100)
    #print(rmses_pso)
    #print(rmses_gn)



    #print(exchanges)

    first = rounds[0][0]

    actual = dist(1, 0)
    simple = calc_simple(first)*c_in_air
    com = calc_complex_tof(first)*c_in_air

    print(actual, simple, com)

    actual = dist(1, 0)
    simple = calc_simple(first,comb_delay=rx_delays[0]+rx_delays[1]+tx_delays[0]+tx_delays[1]) * c_in_air
    com = calc_complex_tof(first, comb_delay=rx_delays[0]+rx_delays[1]+tx_delays[0]+tx_delays[1]) * c_in_air
    print(actual, simple, com)
