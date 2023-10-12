import numpy as np

from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd
import ctypes

from testbed_to_c_vals import  create_inference_matrix



def convert_logged_measurement(val):

    if val > 2**62:
        #this value is likely overflown...
        number = val & 0xFFFFFFFFFFFFFFFF
        signed_number = ctypes.c_longlong(number).value
        val = signed_number

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



def extract_measurements(msg_iter, testbed, src_dev, include_dummy=False, num_ci_drift_avg = 0):
    rounds, raw_measurements, drift_estimations = extract_types(msg_iter, ['raw_measurements', 'drift_estimation'])

    # put all messages into (d, round) sets

    for r in rounds:
        for (d_index, d) in enumerate(testbed.devs):

            if src_dev and d != src_dev:
                continue # we skip as we do not have this data anyway!

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
                    record['tdoa'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[d])-get_dist(testbed.dev_positions[d], testbed.dev_positions[db])

                    msg = raw_measurements.get((d, r), None)

                    if msg is not None and include_dummy == False and 'dummy' in msg and msg['dummy'] == True:
                        continue

                    record['estimated_tof'] = None
                    record['round_dur'] = None
                    record['response_dur'] = None
                    record['tdoa_m'] = None
                    record['estimated_tdoa'] = None

                    if msg is not None and pi < len(msg['measurements']) and msg['measurements'][pi] is not None:
                        record['round_dur'] = msg['measurements'][pi][0]
                        record['response_dur'] = msg['measurements'][pi][1]

                        if msg['measurements'][pi][2] is not None:
                            record['estimated_tof'] = convert_ts_to_m(convert_logged_measurement(msg['measurements'][pi][2]))

                        if len(msg['measurements'][pi]) >= 7 and msg['measurements'][pi][4] != 0:
                            record['tdoa_m'] = msg['measurements'][pi][4]
                            record['estimated_tdoa'] = convert_ts_to_m(convert_logged_measurement(msg['measurements'][pi][5]))

                    msg = drift_estimations.get((d, r), None)

                    record['own_dur_a'] = None
                    record['other_dur_a'] = None
                    record['relative_drift_a'] = None
                    record['relative_drift_a_ci'] = None
                    record['own_dur_b'] = None
                    record['other_dur_b'] = None
                    record['relative_drift_b'] = None
                    record['relative_drift_b_ci'] = None


                    # calculate average drift estimation using carrierintegrators, note that we are using dummy rounds here as well!

                    sum_ci_drift_a = 0.0
                    num_ci_drift_a = 0
                    sum_ci_drift_b = 0.0
                    num_ci_drift_b = 0

                    for old_r in range(r-num_ci_drift_avg, r+1):
                        old_ci_msg = drift_estimations.get((d, old_r), None)
                        if old_ci_msg is None:
                            continue
                        if old_ci_msg['carrierintegrators'][a] != 0:
                            sum_ci_drift_a += ci_to_rd(old_ci_msg['carrierintegrators'][a])
                            num_ci_drift_a += 1

                        if old_ci_msg['carrierintegrators'][b] != 0:
                            sum_ci_drift_b += ci_to_rd(old_ci_msg['carrierintegrators'][b])
                            num_ci_drift_b += 1

                    if num_ci_drift_a > 0:
                        record['relative_drift_a_ci_avg'] = sum_ci_drift_a / num_ci_drift_a

                    if num_ci_drift_b > 0:
                        record['relative_drift_b_ci_avg'] = sum_ci_drift_b / num_ci_drift_b


                    if a == d_index:
                        record['relative_drift_a_ci'] = 1.0
                        record['relative_drift_a_ci_avg'] = 1.0
                    if b == d_index:
                        record['relative_drift_b_ci'] = 1.0
                        record['relative_drift_b_ci_avg'] = 1.0

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

                    if None not in [record['relative_drift_a_ci'], record['relative_drift_b_ci'], record['round_dur'], record['response_dur']]:
                        record['calculated_tof_ci'] = convert_ts_to_m(record['relative_drift_a_ci'] *record['round_dur'] - record['relative_drift_b_ci'] * record['response_dur'])*0.5
                        record['calculated_tof_ci_avg'] = convert_ts_to_m(record['relative_drift_a_ci_avg'] *record['round_dur'] - record['relative_drift_b_ci_avg'] * record['response_dur'])*0.5


                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'], record['response_dur'], record['tdoa_m']]:
                        record['calculated_tdoa'] = convert_ts_to_m(record['relative_drift_a'] *record['round_dur']*0.5 + record['relative_drift_b'] * record['response_dur']*0.5-record['tdoa_m'])

                    if None not in [record['relative_drift_a_ci'], record['relative_drift_b_ci'], record['round_dur'], record['response_dur'], record['tdoa_m']]:
                        record['calculated_tdoa_ci'] = convert_ts_to_m(record['relative_drift_a_ci'] *record['round_dur']*0.5 + record['relative_drift_b_ci'] * record['response_dur']*0.5-record['tdoa_m'])
                        record['calculated_tdoa_ci_avg'] = convert_ts_to_m(record['relative_drift_a_ci_avg'] *record['round_dur']*0.5 + record['relative_drift_b_ci_avg'] * record['response_dur']*0.5-record['tdoa_m'])


                    yield record


def extract_estimations(msg_iter, testbed, src_dev):
    rounds, estimations = extract_types(msg_iter, ['estimation'])

    records = []
    for r in rounds:
        for d in testbed.devs:
            if src_dev and d != src_dev:
                continue # we skip as we do not have this data anyway!

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

                    #print(pi, len(msg['mean_measurements']))
                    if msg is not None and pi < len(msg['mean_measurements']):
                        record['mean_measurement'] = convert_ts_to_m(convert_logged_measurement(msg['mean_measurements'][pi]))

                        if 'var_measurements' in msg:
                            record['var_measurement'] = np.square(convert_ts_to_m(np.sqrt(convert_logged_measurement(msg['var_measurements'][pi]))))
                        else:
                            record['var_measurement'] = None

                        record['est_distance_uncalibrated'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_uncalibrated'][pi]))
                        record['est_distance_factory'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_factory_delays'][pi]))
                        record['est_distance_calibrated'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_estimated_delays'][pi]))

                        if 'tofs_from_filtered_estimated_delays' in msg:
                            record['est_distance_calibrated_filtered'] = convert_ts_to_m(convert_logged_measurement(msg['tofs_from_filtered_estimated_delays'][pi]))
                        else:
                            record['est_distance_calibrated_filtered'] = None

                    else:
                        record['mean_measurement'] = None
                        record['var_measurement'] = None
                        record['est_distance_uncalibrated'] = None
                        record['est_distance_factory'] = None
                        record['est_distance_calibrated'] = None
                        record['est_distance_calibrated_filtered'] = None

                    yield record


def extract_delay_estimates(msg_iter, testbed, src_dev, ignore_pairs=[]):
    rounds, estimations = extract_types(msg_iter, ['estimation'])
    for r in rounds:
        for d in testbed.devs:
            if src_dev and d != src_dev:
                continue  # we skip as we do not have this data anyway!

            msg = estimations.get((d, r), None)

            if msg:
                record = {}
                record['delays_from_measurements'] = []
                record['delays_from_measurements_rounded'] = []
                record['factory_delays'] = []
                record['factory_delays_diff'] = []

                for dm in msg['delays_from_measurements']:
                    record['delays_from_measurements'].append(convert_logged_measurement(dm))
                    record['delays_from_measurements_rounded'].append(round(convert_logged_measurement(dm)))

                for (i,d) in enumerate(testbed.devs):
                    record['factory_delays'].append(testbed.factory_delays[d]-16450)
                    record['factory_delays_diff'].append(record['delays_from_measurements_rounded'][i]-record['factory_delays'][i])

                M = create_inference_matrix(len(testbed.devs), ignored_pairs=ignore_pairs)

                measured = np.array([convert_logged_measurement(x) for x in msg['mean_measurements']])

                actual = np.zeros(shape=round(len(testbed.devs)*(len(testbed.devs)-1)/2))

                for (a, da) in enumerate(testbed.devs):
                    for (b, db) in enumerate(testbed.devs):
                        if a > b:
                            actual[pair_index(a,b)] = convert_m_to_ts(get_dist(testbed.dev_positions[da], testbed.dev_positions[db]))

                diffs = measured - actual

                record['python_delays'] = np.matmul(M, 2*diffs)
                record['python_delays_rounded'] = [round(x) for x in record['python_delays']]

                record['python_delays_diff'] = record['python_delays_rounded']-np.array(record['factory_delays'])
                record['mse'] = np.sqrt(np.mean(np.square(record['python_delays_diff'])))
                record['mae'] = np.mean(np.abs(record['python_delays_diff']))

                yield record


def gen_measurements_from_testbed_run(testbed, run, src_dev=None, include_dummy=False, num_ci_drift_avg=0):
    logfile = "data/{}/{}.log".format(testbed.name, run)

    with open(logfile) as f:
        yield from extract_measurements(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed, src_dev=src_dev, include_dummy=include_dummy, num_ci_drift_avg=num_ci_drift_avg)


def gen_estimations_from_testbed_run(testbed, run, src_dev=None):
    logfile = "data/{}/{}.log".format(testbed.name, run)
    with open(logfile) as f:
        yield from extract_estimations(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed, src_dev=src_dev)

def gen_delay_estimates_from_testbed_run(testbed, run, src_dev=None, ignore_pairs=[]):
    logfile = "data/{}/{}.log".format(testbed.name, run)
    with open(logfile) as f:
        yield from extract_delay_estimates(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed, src_dev=src_dev, ignore_pairs=ignore_pairs)


import pandas as pd

def extract_tdma_twr(testbed, run, tdoa_src_dev_number=None):
    logfile = "data/{}/{}.log".format(testbed.name, run)

    rx_events = []
    tx_events = []

    with open(logfile) as f:
        msg_gen = testbed.parse_messages_from_lines(f)
        for (_, _, x) in msg_gen:
            e = x['event']
            if e == 'rx':
                rx_events.append(x)
            elif e == 'tx':
                tx_events.append(x)
            elif e == 'init':
                pass

    rx_df = pd.DataFrame.from_records(rx_events)
    tx_df = pd.DataFrame.from_records(tx_events)
    max_round = rx_df['rx_round'].max()

    def extract_data(r, a, b, passive):
        # we extract data for initiator a and responder b (numbers not device names!)

        init_slot = a*(len(testbed.devs)-1)*3+b*3

        if a <= b:
            init_slot -= 3 # we save one exchange because a does not exchange it with itself...

        respond_slot = init_slot + 1
        final_slot = respond_slot + 1

        def first_or_none(l):
            return next(iter(l), None)

        init_tx = first_or_none(tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == init_slot) & (tx_df["own_number"] == a)].to_dict(orient='records'))
        init_rx = first_or_none(rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == init_slot) & (rx_df["own_number"] == b)].to_dict(orient='records'))
        init_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == init_slot) & (rx_df["own_number"] == passive)].to_dict(orient='records'))

        respond_tx = first_or_none(tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == respond_slot) & (tx_df["own_number"] == b)].to_dict(orient='records'))
        respond_rx = first_or_none(rx_df[(rx_df["rx_round"] == r) &(rx_df["rx_slot"] == respond_slot) & (rx_df["own_number"] == a)].to_dict(orient='records'))
        respond_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) &(rx_df["rx_slot"] == respond_slot) & (rx_df["own_number"] == passive)].to_dict(orient='records'))

        final_tx = first_or_none(tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == final_slot) & (tx_df["own_number"] == a)].to_dict(orient='records'))
        final_rx = first_or_none(rx_df[(rx_df["rx_round"] == r) &(rx_df["rx_slot"] == final_slot) & (rx_df["own_number"] == b)].to_dict(orient='records'))
        final_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) &(rx_df["rx_slot"] == final_slot) & (rx_df["own_number"] == passive)].to_dict(orient='records'))


        assert init_rx is None or init_rx['rx_number'] == a
        assert init_rx_passive is None or init_rx_passive['rx_number'] == a

        assert respond_rx is None or respond_rx['rx_number'] == b
        assert respond_rx_passive is None or respond_rx_passive['rx_number'] == b

        assert final_rx is None or final_rx['rx_number'] == a
        assert final_rx_passive is None or final_rx_passive['rx_number'] == a


        ret = {
            'init_tx': init_tx,
            'init_rx': init_rx,
            'init_rx_passive': init_rx_passive,
            'respond_tx': respond_tx,
            'respond_rx': respond_rx,
            'respond_rx_passive': respond_rx_passive,
            'final_tx': final_tx,
            'final_rx': final_rx,
            'final_rx_passive': final_rx_passive
        }

        return ret

    if tdoa_src_dev_number:
        tdoa_src_dev = testbed.devs[tdoa_src_dev_number]
    else:
        tdoa_src_dev = None

    for r in range(0, max_round+1):

            for (a, da) in enumerate(testbed.devs):
                for (b, db) in enumerate(testbed.devs):

                    if a == b:
                        continue

                    record = {}

                    record['round'] = int(r)
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a,b)
                    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])
                    record['tdoa'] = None

                    data = extract_data(r, a, b, tdoa_src_dev_number)

                    # we check if data contains any None values, if so, we drop this exchange
                    def dur(start, end):
                        if None in [start, end]:
                            return None

                        if end <= start:
                            end += 0xFFFFFFFFFF+1 # we handle overflow here

                        return end - start

                    round_a = dur((data.get('init_tx', {}) or {}).get('tx_ts', None), (data.get('respond_rx', {}) or {}).get('bias_corrected_rx_ts', None))
                    delay_b = dur((data.get('init_rx', {}) or {}).get('bias_corrected_rx_ts', None), (data.get('respond_tx', {}) or {}).get('tx_ts', None))
                    delay_a = dur((data.get('respond_rx', {}) or {}).get('bias_corrected_rx_ts', None), (data.get('final_tx', {}) or {}).get('tx_ts', None))
                    round_b = dur((data.get('respond_tx', {}) or {}).get('tx_ts', None), (data.get('final_rx', {}) or {}).get('bias_corrected_rx_ts', None))

                    record['round_a'] = round_a
                    record['delay_b'] = delay_b
                    record['delay_a'] = delay_a
                    record['round_b'] = round_b


                    if None not in [round_a, delay_b, delay_a, round_b]:
                        relative_drift = float(round_a+delay_a)/float(round_b+delay_b)
                        twr_tof = convert_ts_to_m((round_a - relative_drift * delay_b)*0.5)

                        record['relative_drift'] = relative_drift
                        record['twr_tof'] = twr_tof


                        if tdoa_src_dev:
                            record['tdoa'] = get_dist(testbed.dev_positions[da],
                                                      testbed.dev_positions[tdoa_src_dev]) - get_dist(
                                testbed.dev_positions[tdoa_src_dev], testbed.dev_positions[db])
                            record['tdoa_device'] = tdoa_src_dev_number

                            passive_m_a = dur((data.get('init_passive_rx', {}) or {}).get('rx_ts', None), (data.get('respond_passive_rx', {}) or {}).get('bias_corrected_rx_ts', None))
                            passive_m_b = dur((data.get('respond_passive_rx', {}) or {}).get('rx_ts', None), (data.get('final_passive_rx', {}) or {}).get('bias_corrected_rx_ts', None))
                            record['passive_m_a'] = passive_m_a
                            record['passive_m_b'] = passive_m_b
                    yield record


# import testbed.trento_b as trento_b
# it = extract_tdma_twr(trento_b, 'job_tdma')
#
# df = pd.DataFrame.from_records(it)
#
# #print((df[df['pair'] == '12-6']).to_string())
# #exit()
#
# gb = df.groupby('pair')
#
# print("MEAN Error")
# print(((gb['twr_tof']).mean() - gb['dist'].mean()).to_string())
#
#
# print("STD")
# print((gb['twr_tof'].std()*100).to_string())
#
# print("Count")
# print((gb['twr_tof'].count()).to_string())




# TODO comment in!
# from testbed import trento_a, trento_b, lille
#
# print("TRENTO A All Pairs")
# res = list(gen_delay_estimates_from_testbed_run(trento_a, run='job_fixed', src_dev=None, ignore_pairs=[]))[0]
# print(res['mae'])
# trento_a_unfiltered = res
#
# print("TRENTO B All Pairs")
# res = list(gen_delay_estimates_from_testbed_run(trento_b, run='job_fixed', src_dev=None, ignore_pairs=[]))[0]
# print(res['mae'])
#
# print("LILLE All Pairs")
# res = list(gen_delay_estimates_from_testbed_run(lille, run='job_fixed', src_dev=None, ignore_pairs=[]))[0]
# print(res['mae'])
#
# print("TRENTO A Filtered")
# res = list(gen_delay_estimates_from_testbed_run(trento_a, run='job_fixed', src_dev=None, ignore_pairs=[(6,3)]))[0]
# print(res['mae'])
# trento_a_filtered = res
#
# print("LILLE Filtered")
# res = list(gen_delay_estimates_from_testbed_run(lille, run='job_fixed', src_dev=None, ignore_pairs=[(7,1), (4, 2)]))[0]
# print(res['mae'])
#
#
#
# exp = trento_a_unfiltered
#
# for i in range(0, 7):
#     est = exp['delays_from_measurements_rounded'][i]*0.47
#     fact = exp['factory_delays'][i]*0.47
#     diff = est-fact
#     print("{} & {:2.2f} & {:2.2f} & {:2.2f} \\\\ \\hline".format(i+1, est, fact, diff))



#1 & XX & 10.34 & XX \\  \hline