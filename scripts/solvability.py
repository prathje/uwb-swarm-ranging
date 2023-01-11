import numpy as np


# Speed of light in air
c_in_air = 1.0 #299702547.236

def get_dof(NUM_NODES, given_delays, given_distances, with_tdoa):
    positions = np.random.uniform(low=0.0, high=100.0, size=(NUM_NODES,2))

    def dist(a,b):
        return np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))

    def tof(a,b):
        return dist(a,b)/c_in_air

    # we need separate tx and rx_delays for the message exchange
    combined_delays = np.random.uniform(low=0.0, high=0.005, size=NUM_NODES)


    # we simulate a ranging exchange between all 3 device combinations

    exchanges = {}

    for i in range(NUM_NODES):
        for r in range(NUM_NODES):

            if i == r:
                continue

            response_delay = 0.1  # THIS needs to be the same for nownp.random.uniform(0.0, 0.1)
            round_time = 2 * tof(i, r) + response_delay

            assert response_delay >= 0
            assert round_time >= response_delay

            measured_response_delay = response_delay - combined_delays[r]
            measured_round_time = round_time + combined_delays[i]

            computed_tof = (measured_round_time - measured_response_delay) / 2.0


            measured_eavesdrop_times = {}
            computed_tdoas = {}

            for l in range(NUM_NODES):
                if l == i or l == r:
                    continue

                eavesdrop_time = tof(i, r) + response_delay + tof(r, l) - tof(i, l)
                assert eavesdrop_time >= 0
                measured_eavesdrop_times[l] = eavesdrop_time # no noise for now (delays are +-0)
                computed_tdoas[l] = computed_tof + measured_response_delay - eavesdrop_time

            exchanges[(i, r)] = {
                'response_delay': response_delay,
                'round_time': round_time,
                'measured_response_delay': measured_response_delay,
                'measured_round_time': measured_round_time,
                'measured_eavesdrop_times': measured_eavesdrop_times,
                'computed_tdoas': computed_tdoas,
                'computed_tof': computed_tof
            }



    factor_rows = []
    target_rows = []


    factor_count = 0

    delay_indices = {}
    distance_indices = {}

    for i in range(NUM_NODES):
        # add factors for the delays
        delay_indices[i] = factor_count
        factor_count += 1

    for a in range(NUM_NODES):
        for b in range(NUM_NODES):
            if a < b:
                distance_indices[(a,b)] = factor_count
                distance_indices[(b,a)] = factor_count
                factor_count += 1

    # extract equations from each exchange
    for (i,r) in exchanges:
        ex = exchanges[(i,r)]

        # first the Tof equation
        factors = np.zeros(shape=factor_count)

        factors[delay_indices[i]] = 1
        factors[delay_indices[r]] = 1
        factors[distance_indices[(i, r)]] = 2
        target = 2*ex['computed_tof']

        factor_rows.append(factors)
        target_rows.append(target)


        # NOW TDoA Stuff
        if with_tdoa:
            for l in ex['measured_eavesdrop_times']:
                # then the TDoA one:

                factors = np.zeros(shape=factor_count)

                factors[delay_indices[i]] = 1
                factors[delay_indices[r]] = -1
                factors[distance_indices[(i, l)]] = 2
                factors[distance_indices[(r, l)]] = -2

                target = 2 * ex['computed_tdoas'][l]

                factor_rows.append(factors)
                target_rows.append(target)



                # then the NEW TDoA one:
                factors = np.zeros(shape=factor_count)

                factors[delay_indices[i]] = 1
                factors[distance_indices[(i, r)]] = 1
                factors[distance_indices[(i, l)]] = 1
                factors[distance_indices[(r, l)]] = -1

                target = ex['measured_round_time']-ex['measured_eavesdrop_times'][l]

                factor_rows.append(factors)
                target_rows.append(target)

                # then the NEW NEW TDoA one:

                factors = np.zeros(shape=factor_count)

                factors[delay_indices[r]] = 1
                factors[distance_indices[(i, r)]] = 1
                factors[distance_indices[(i, l)]] = -1
                factors[distance_indices[(r, l)]] = 1

                target = ex['measured_eavesdrop_times'][l] - ex['measured_response_delay']

                factor_rows.append(factors)
                target_rows.append(target)

            # Now TDoA Difference Part 2
            for lx in ex['measured_eavesdrop_times']:
                for ly in ex['measured_eavesdrop_times']:
                    if lx != ly:

                        factors = np.zeros(shape=factor_count)
                        factors[distance_indices[(r, lx)]] = 1
                        factors[distance_indices[(r, ly)]] = -1
                        factors[distance_indices[(i, lx)]] = -1
                        factors[distance_indices[(i, ly)]] = 1

                        target = ex['measured_eavesdrop_times'][lx] - ex['measured_eavesdrop_times'][ly]

                        factor_rows.append(factors)
                        target_rows.append(target)


    # since this is rank defficient, we have to add more values!




    #    given_delays = range(NUM_GIVEN_DELAYS)
    #    given_distances = list(exchanges.keys())[:NUM_GIVEN_DISTANCES]



    # given_delays = []# [0,1,2]
    # given_distances = [] # [(0,1),(0,2), (1,2)]
    #

    for i in given_delays:
        # # Add Delay of node 0
        factors = np.zeros(shape=factor_count)
        factors[delay_indices[i]] = 1
        target = combined_delays[i]
        factor_rows.append(factors)
        target_rows.append(target)

    for (i,r) in given_distances:
        # # Add tof of nodes i and r
        factors = np.zeros(shape=factor_count)
        factors[distance_indices[(i,r)]] = 1
        target = tof(i,r)
        factor_rows.append(factors)
        target_rows.append(target)

    A = np.array(factor_rows)
    b = np.array(target_rows)

    return factor_count - (np.linalg.matrix_rank(A))


NUM_NODES = 6

import itertools

all_dists = []

for a in range(NUM_NODES):
    for b in range(NUM_NODES):
        if a < b:
            all_dists.append((a,b))


stats = {}

for cd in range(NUM_NODES+1):
    for d in range(len(all_dists)+1):
        stats[(cd,d)] = []
        for delays in itertools.combinations(range(NUM_NODES), cd):
            for distances in itertools.combinations(all_dists, d):
                dof = get_dof(NUM_NODES, delays, distances, False)
                dof_with_tdoa = dof #get_dof(NUM_NODES, delays, distances, True)

                if dof != dof_with_tdoa:
                    print("DIFFERENCE")
                stats[(cd, d)].append(dof)

        xs = np.array(stats[(cd, d)])
        print((cd, d), np.mean(xs), np.min(xs), np.max(xs))