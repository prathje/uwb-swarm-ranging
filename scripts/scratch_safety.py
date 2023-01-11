import numpy as np


# Speed of light in air
c_in_air = 1.0 #299702547.236

NUM_NODES = 3
positions = [(0,0), (100,0), (0,100)]

def dist(a,b):
    return np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))

def tof(a,b):
    return dist(a,b)/c_in_air

# we need separate tx and rx_delays for the message exchange
combined_delays = [0.00338113, 0.0030542,  0.00109631] #np.random.uniform(0.0, 0.01, NUM_NODES)

# we simulate a ranging exchange between all 3 device combinations

exchanges = {}

for i in range(NUM_NODES):
    for r in range(NUM_NODES):
        for l in range(NUM_NODES):

            if len(set([i,r,l])) != 3:
                continue

            response_delay = 0.1 # fixed for now!  TODO! np.random.uniform(0.0, 0.1)
            round_time = 2*tof(i,r) + response_delay
            eavesdrop_time = tof(i,r) + response_delay + tof(r,l) - tof(i,l)

            assert response_delay >= 0
            assert eavesdrop_time >= response_delay
            assert round_time >= response_delay

            measured_response_delay = response_delay - combined_delays[r]
            measured_round_time = round_time + combined_delays[i]
            measured_eavesdrop_time = eavesdrop_time    # no noise for now (delays are +-0)

            computed_tof = (measured_round_time-measured_response_delay)/2.0
            computed_tdoa = computed_tof + measured_response_delay - measured_eavesdrop_time

            exchanges[(i,r,l)] = {
                'response_delay': response_delay,
                'round_time': round_time,
                'eavesdrop_time': eavesdrop_time,
                'measured_response_delay': measured_response_delay,
                'measured_round_time': measured_round_time,
                'measured_eavesdrop_time': measured_eavesdrop_time,
                'computed_tof': computed_tof,
                'computed_tdoa': computed_tdoa,
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
for (i,r,l) in exchanges:
    ex = exchanges[(i,r,l)]

    # first the Tof equation
    factors = np.zeros(shape=factor_count)

    factors[delay_indices[i]] = 1
    factors[delay_indices[r]] = 1
    factors[distance_indices[(i, r)]] = 2
    target = 2*ex['computed_tof']

    factor_rows.append(factors)
    target_rows.append(target)

    # then the TDoA one:

    factors = np.zeros(shape=factor_count)

    factors[delay_indices[i]] = 1
    factors[delay_indices[r]] = -1
    factors[distance_indices[(i, l)]] = 2
    factors[distance_indices[(r, l)]] = -2

    target = 2 * ex['computed_tdoa']

    factor_rows.append(factors)
    target_rows.append(target)


# since this is rank defficient, we have to add more values!

# Add Delay of node 0
factors = np.zeros(shape=factor_count)
factors[delay_indices[0]] = 1
target = combined_delays[0]
factor_rows.append(factors)
target_rows.append(target)

# Add Delay of node 1
factors = np.zeros(shape=factor_count)
factors[delay_indices[1]] = 1
target = combined_delays[1]
factor_rows.append(factors)
target_rows.append(target)

# Add Delay of node 2
factors = np.zeros(shape=factor_count)
factors[delay_indices[2]] = 1
target = combined_delays[2]
factor_rows.append(factors)
target_rows.append(target)

A = np.array(factor_rows)
b = np.array(target_rows)

print(A)

print("Rank")
print(np.linalg.matrix_rank(A))
#print(A)
#print(b)
solution = np.linalg.lstsq(A, b)

print("Delays")
print(combined_delays)
print("Estimated Delays")
print(solution[0:3])