from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd

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



def extract_measurements(msg_iter, testbed):
    rounds, raw_measurements, drift_estimations = extract_types(msg_iter, ['raw_measurements', 'drift_estimation'])

    # put all messages into (d, round) sets

    for r in rounds:
        for d in testbed.devs:
            for (a, da) in enumerate(testbed.devs):
                for (b, db) in enumerate(testbed.devs):
                    if b <= a:
                        continue
                    pi = pair_index(a,b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a,b)
                    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])

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
                    record['relative_drift_a_ci'] = None
                    record['own_dur_b'] = None
                    record['other_dur_b'] = None
                    record['relative_drift_b'] = None
                    record['relative_drift_b_ci'] = None

                    if msg is not None:

                        if 'carrierintegrators' in msg:
                            if msg['carrierintegrators'][a] != 0:
                                record['relative_drift_a_ci'] = ci_to_rd(msg['carrierintegrators'][a])
                            if msg['carrierintegrators'][b] != 0:
                                record['relative_drift_b_ci'] = ci_to_rd(msg['carrierintegrators'][b])

                        if msg['durations'][a] is not None:
                            record['own_dur_a'] = msg['durations'][a][0]
                            record['other_dur_a'] = msg['durations'][a][1]
                            if record['own_dur_a'] != 0 and record['other_dur_a'] != 0:
                                record['relative_drift_a'] = float(record['own_dur_a']) / float(record['other_dur_a'])

                        if msg['durations'][b] is not None:
                            record['own_dur_b'] = msg['durations'][b][0]
                            record['other_dur_b'] = msg['durations'][b][1]
                            if record['own_dur_b'] != 0 and record['other_dur_b'] != 0:
                                record['relative_drift_b'] = float(record['own_dur_b']) / float(record['other_dur_b'])

                    record['calculated_tof'] = None
                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'], record['response_dur']]:
                        record['calculated_tof'] = convert_ts_to_m(record['relative_drift_a'] *record['round_dur'] - record['relative_drift_b'] * record['response_dur'])*0.5

                    yield record


def extract_estimations(msg_iter, testbed):
    rounds, estimations = extract_types(msg_iter, ['estimation'])

    records = []
    for r in rounds:
        for d in testbed.devs:
            for (a, da) in enumerate(testbed.devs):
                for (b, db) in enumerate(testbed.devs):
                    if b <= a:
                        continue

                    pi = pair_index(a,b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a,b)
                    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])

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


def gen_measurements_from_testbed_run(testbed, run, src_dev=0):
    logfile = "data/{}/{}.log".format(testbed.name, run)

    with open(logfile) as f:
        yield from extract_measurements(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed)


def gen_estimations_from_testbed_run(testbed, run, src_dev=0):
    logfile = "data/{}/{}.log".format(testbed.name, run)
    with open(logfile) as f:
        yield from extract_estimations(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed)