import numpy as np

from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd
import ctypes

from testbed_to_c_vals import create_inference_matrix

DS_OVERFLOW_VAL = 0xFFFFFFFFFF+1


def convert_logged_measurement(val):
    if val > 2 ** 62:
        # this value is likely overflown...
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


def extract_measurements(msg_iter, testbed, src_dev, include_dummy=False, num_ci_drift_avg=0):
    rounds, raw_measurements, drift_estimations = extract_types(msg_iter, ['raw_measurements', 'drift_estimation'])

    # put all messages into (d, round) sets

    for r in rounds:
        for (d_index, d) in enumerate(testbed.devs):

            if src_dev and d != src_dev:
                continue  # we skip as we do not have this data anyway!

            for (a, da) in enumerate(testbed.devs):
                for (b, db) in enumerate(testbed.devs):
                    if b <= a:
                        continue

                    pi = pair_index(a, b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a, b)
                    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])
                    record['tdoa'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[d]) - get_dist(
                        testbed.dev_positions[d], testbed.dev_positions[db])

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
                            record['estimated_tof'] = convert_ts_to_m(
                                convert_logged_measurement(msg['measurements'][pi][2]))

                        if len(msg['measurements'][pi]) >= 7 and msg['measurements'][pi][4] != 0:
                            record['tdoa_m'] = msg['measurements'][pi][4]
                            record['estimated_tdoa'] = convert_ts_to_m(
                                convert_logged_measurement(msg['measurements'][pi][5]))

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

                    for old_r in range(r - num_ci_drift_avg, r + 1):
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

                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'],
                                    record['response_dur']]:
                        record['calculated_tof'] = convert_ts_to_m(
                            record['relative_drift_a'] * record['round_dur'] - record['relative_drift_b'] * record[
                                'response_dur']) * 0.5

                    if None not in [record['relative_drift_a_ci'], record['relative_drift_b_ci'], record['round_dur'],
                                    record['response_dur']]:
                        record['calculated_tof_ci'] = convert_ts_to_m(
                            record['relative_drift_a_ci'] * record['round_dur'] - record['relative_drift_b_ci'] *
                            record['response_dur']) * 0.5
                        record['calculated_tof_ci_avg'] = convert_ts_to_m(
                            record['relative_drift_a_ci_avg'] * record['round_dur'] - record[
                                'relative_drift_b_ci_avg'] * record['response_dur']) * 0.5

                    if None not in [record['relative_drift_a'], record['relative_drift_b'], record['round_dur'],
                                    record['response_dur'], record['tdoa_m']]:
                        record['calculated_tdoa'] = convert_ts_to_m(
                            record['relative_drift_a'] * record['round_dur'] * 0.5 + record['relative_drift_b'] *
                            record['response_dur'] * 0.5 - record['tdoa_m'])

                    if None not in [record['relative_drift_a_ci'], record['relative_drift_b_ci'], record['round_dur'],
                                    record['response_dur'], record['tdoa_m']]:
                        record['calculated_tdoa_ci'] = convert_ts_to_m(
                            record['relative_drift_a_ci'] * record['round_dur'] * 0.5 + record['relative_drift_b_ci'] *
                            record['response_dur'] * 0.5 - record['tdoa_m'])
                        record['calculated_tdoa_ci_avg'] = convert_ts_to_m(
                            record['relative_drift_a_ci_avg'] * record['round_dur'] * 0.5 + record[
                                'relative_drift_b_ci_avg'] * record['response_dur'] * 0.5 - record['tdoa_m'])

                    yield record


def extract_estimations(msg_iter, testbed, src_dev):
    rounds, estimations = extract_types(msg_iter, ['estimation'])

    records = []
    for r in rounds:
        for d in testbed.devs:
            if src_dev and d != src_dev:
                continue  # we skip as we do not have this data anyway!

            for (a, da) in enumerate(testbed.devs):
                for (b, db) in enumerate(testbed.devs):
                    if b <= a:
                        continue

                    pi = pair_index(a, b)

                    record = {}

                    record['round'] = int(r)
                    record['device'] = d
                    record['initiator'] = a
                    record['responder'] = b
                    record['pair'] = "{}-{}".format(a, b)
                    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])

                    msg = estimations.get((d, r), None)

                    # print(pi, len(msg['mean_measurements']))
                    if msg is not None and pi < len(msg['mean_measurements']):
                        record['mean_measurement'] = convert_ts_to_m(
                            convert_logged_measurement(msg['mean_measurements'][pi]))

                        if 'var_measurements' in msg:
                            record['var_measurement'] = np.square(
                                convert_ts_to_m(np.sqrt(convert_logged_measurement(msg['var_measurements'][pi]))))
                        else:
                            record['var_measurement'] = None

                        record['est_distance_uncalibrated'] = convert_ts_to_m(
                            convert_logged_measurement(msg['tofs_uncalibrated'][pi]))
                        record['est_distance_factory'] = convert_ts_to_m(
                            convert_logged_measurement(msg['tofs_from_factory_delays'][pi]))
                        record['est_distance_calibrated'] = convert_ts_to_m(
                            convert_logged_measurement(msg['tofs_from_estimated_delays'][pi]))

                        if 'tofs_from_filtered_estimated_delays' in msg:
                            record['est_distance_calibrated_filtered'] = convert_ts_to_m(
                                convert_logged_measurement(msg['tofs_from_filtered_estimated_delays'][pi]))
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

                for (i, d) in enumerate(testbed.devs):
                    record['factory_delays'].append(testbed.factory_delays[d] - 16450)
                    record['factory_delays_diff'].append(
                        record['delays_from_measurements_rounded'][i] - record['factory_delays'][i])

                M = create_inference_matrix(len(testbed.devs), ignored_pairs=ignore_pairs)

                measured = np.array([convert_logged_measurement(x) for x in msg['mean_measurements']])

                actual = np.zeros(shape=round(len(testbed.devs) * (len(testbed.devs) - 1) / 2))

                for (a, da) in enumerate(testbed.devs):
                    for (b, db) in enumerate(testbed.devs):
                        if a > b:
                            actual[pair_index(a, b)] = convert_m_to_ts(
                                get_dist(testbed.dev_positions[da], testbed.dev_positions[db]))

                diffs = measured - actual

                record['python_delays'] = np.matmul(M, 2 * diffs)
                record['python_delays_rounded'] = [round(x) for x in record['python_delays']]

                record['python_delays_diff'] = record['python_delays_rounded'] - np.array(record['factory_delays'])
                record['mse'] = np.sqrt(np.mean(np.square(record['python_delays_diff'])))
                record['mae'] = np.mean(np.abs(record['python_delays_diff']))

                yield record


def gen_measurements_from_testbed_run(testbed, run, src_dev=None, include_dummy=False, num_ci_drift_avg=0):
    logfile = "data/{}/{}.log".format(testbed.name, run)

    with open(logfile) as f:
        yield from extract_measurements(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed,
                                        src_dev=src_dev, include_dummy=include_dummy, num_ci_drift_avg=num_ci_drift_avg)


def gen_estimations_from_testbed_run(testbed, run, src_dev=None):
    logfile = "data/{}/{}.log".format(testbed.name, run)
    with open(logfile) as f:
        yield from extract_estimations(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed,
                                       src_dev=src_dev)


def gen_delay_estimates_from_testbed_run(testbed, run, src_dev=None, ignore_pairs=[]):
    logfile = "data/{}/{}.log".format(testbed.name, run)
    with open(logfile) as f:
        yield from extract_delay_estimates(testbed.parse_messages_from_lines(f, src_dev=src_dev), testbed=testbed,
                                           src_dev=src_dev, ignore_pairs=ignore_pairs)


import pandas as pd


def gen_round_events(testbed, run, iter_per_device=True):
    logfile = "data/{}/{}.log".format(testbed.name, run)

    def gen_round_events_for_iter(it, filter_dev=None):
        it_rx_events = []
        it_tx_events = []
        it_round = None

        for log_ts, dev, x in it:
            if filter_dev is not None and filter_dev != dev:
                continue

            e = x['event']

            if e == 'rx':
                e_r = int(x['rx_round'])
            elif e == 'tx':
                e_r = int(x['tx_round'])
            else:
                continue

            if it_round is None:
                it_round = e_r

            if it_round < e_r:
                yield it_round, it_rx_events, it_tx_events
                it_rx_events = []
                it_tx_events = []
                it_round = e_r

            if e == 'rx':
                it_rx_events.append(x)
            elif e == 'tx':
                it_tx_events.append(x)
            elif e == 'init':
                pass
        if len(it_rx_events) > 0 or len(it_tx_events) > 0:
            yield it_round, it_rx_events, it_tx_events

    with open(logfile) as f:
        msg_gen = testbed.parse_messages_from_lines(f)

        if iter_per_device:
            import itertools
            it_copies = itertools.tee(msg_gen, len(testbed.devs))
            dev_iters = [gen_round_events_for_iter(it, filter_dev=d) for d, it in zip(testbed.devs, it_copies)]

            round_events = [next(di, (None, [], [])) for di in dev_iters]
            while True:
                min_round = min([re[0] for re in round_events if re[0] is not None], default=None)

                if min_round is None:
                    break  # we are done

                all_rx_events = []
                all_tx_events = []

                for i, re in enumerate(round_events):
                    if re[0] == min_round:
                        all_rx_events += re[1]
                        all_tx_events += re[2]
                        round_events[i] = next(dev_iters[i], (None, [], [])) #advance this it
                    else:
                        print("min_round", min_round, i, re[0])

                yield min_round, all_rx_events, all_tx_events
        else:
            yield from gen_round_events_for_iter(msg_gen)

def gen_all_rx_events(testbed, run, skip_to_round=None, until_round=None):
    round_gen = gen_round_events(testbed, run)

    for (r, rx_events, tx_events) in round_gen:
        if skip_to_round is not None and r < skip_to_round:
            continue
        if until_round is not None and r > until_round:
            break

        for rx_event in rx_events:
            yield rx_event


def estimate_rx_noise_using_cfo(testbed, run, bias_corrected=True, skip_to_round = 0, up_to_round = None):

    def extract_rx_tx_pairs(rx_df, tx_df, transmitter, receiver):
        rel_rx = rx_df[(rx_df['own_number'] == receiver) & (rx_df['rx_number'] == transmitter)]

        last_rx_ts = None
        last_tx_ts = None

        for index, row in rel_rx.iterrows():

            poss_txs = tx_df[(tx_df['own_number'] == transmitter) & (tx_df['tx_round'] == row['rx_round']) & (tx_df['tx_slot'] == row['rx_slot'])]

            if len(poss_txs.index) >= 1:
                rel_tx = poss_txs.iloc[0]
            else:
                continue    # we could not find the relevant transmission!

            rx_ts = row['bias_corrected_rx_ts' if bias_corrected else 'rx_ts']
            tx_ts = rel_tx['tx_ts']

            # correct for overflowing timestamps
            if last_rx_ts is None:
                last_rx_ts = rx_ts
            elif last_rx_ts > rx_ts:
                rx_ts += DS_OVERFLOW_VAL

            if last_tx_ts is None:
                last_tx_ts = tx_ts
            elif last_tx_ts > tx_ts:
                tx_ts += DS_OVERFLOW_VAL

            d = {
                'tx_number': rel_tx['own_number'],
                'rx_number': row['own_number'],
                'round': row['rx_round'],
                'slot': row['rx_slot'],
                'rx_ts': rx_ts,
                'tx_ts': tx_ts,
                'rx_ci': row['ci']
            }

            yield d

    def estimate_noise_std_with_lls(pairs):
        df = pd.DataFrame.from_records(pairs)
        if len(df.index) > 0:
            # rx_ts = tx_ts * alpha + beta + eps_rx

            #df = df[df['slot'] <= 100]
            #df = df.head(10)

            # build coefficient matrix
            coeff = np.asarray([[r['tx_ts'], -1] for (i, r) in df.iterrows()])
            ordinate = df['rx_ts'].to_numpy()

            x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)
            sample_variance = sum_of_squared_residuals / (len(df.index) - 1)

            mean_rd = ci_to_rd(df['rx_ci'].median())
            print("CFO vs estimate", x, mean_rd)
            print(len(df.index), x, sample_variance, convert_ts_to_m(np.sqrt(sample_variance)))

            #print(df)

            return convert_ts_to_m(np.sqrt(sample_variance[0]))

        return None


    import math
    def estimate_noise_std_with_lls_grouped(pairs, group_size=5):
        df = pd.DataFrame.from_records(pairs)
        if len(df.index) > 0:

            # we build the respectice matrices
            coeff = np.asarray([[r['tx_ts'], -1] for (i, r) in df.iterrows()])
            ordinate = df['rx_ts'].to_numpy()

            num_groups = math.ceil(len(df.index)/group_size)

            group_coeff_list = np.array_split(coeff, num_groups)
            ordinate_list = np.array_split(ordinate, num_groups)

            overall_sum_of_squared_residuals = 0.0
            overall_num = 0

            for g_coeff, g_ord in zip(group_coeff_list, ordinate_list):
                if len(g_coeff) >= 4:
                    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(g_coeff, g_ord, rcond=-1)
                    group_sample_variance = sum_of_squared_residuals / (len(g_coeff) - 1)

                    overall_sum_of_squared_residuals += sum_of_squared_residuals[0]
                    overall_num += len(g_coeff)

            sample_variance = overall_sum_of_squared_residuals / (overall_num - 1)

            #print(len(df.index), convert_ts_to_m(np.sqrt(sample_variance)))

            #print(df)

            return convert_ts_to_m(np.sqrt(sample_variance))

        return None

    def estimate_noise_std_with_lls_grouped_median(pairs, group_size=20):
        df = pd.DataFrame.from_records(pairs)
        if len(df.index) > 0:

            # we build the respectice matrices
            coeff = np.asarray([[r['tx_ts'], -1] for (i, r) in df.iterrows()])
            ordinate = df['rx_ts'].to_numpy()

            num_groups = math.ceil(len(df.index)/group_size)

            group_coeff_list = np.array_split(coeff, num_groups)
            ordinate_list = np.array_split(ordinate, num_groups)

            vars = []

            for g_coeff, g_ord in zip(group_coeff_list, ordinate_list):
                if len(g_coeff) >= 4:
                    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(g_coeff, g_ord, rcond=-1)
                    group_sample_variance = sum_of_squared_residuals / (len(g_coeff) - 2)

                    vars.append(group_sample_variance)

            sample_variance = np.median(np.asarray(vars))

            #print(len(df.index), convert_ts_to_m(np.sqrt(sample_variance)))

            #print(df)

            return convert_ts_to_m(np.sqrt(sample_variance))

        return None

    def estimate_noise_std_with_cfo_mean(pairs):

        df = pd.DataFrame.from_records(pairs)
        if len(df.index) > 0:
            df = df.head(5)
            mean_rd = ci_to_rd(df['rx_ci'].median())

            coeff = np.asarray([[-1] for (i, r) in df.iterrows()])
            ordinate = np.asarray([[r['tx_ts']*mean_rd-r['rx_ts']] for (i, r) in df.iterrows()])

            x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

            sample_variance = sum_of_squared_residuals / (len(df.index) - 1)

            sample_std = convert_ts_to_m(np.sqrt(sample_variance[0]))

            print(sample_std)
            return sample_std
        return None

    rows_list = []
    for (r, rx_events, tx_events) in gen_round_events(testbed, run):


        if skip_to_round is not None and r < skip_to_round:
            continue
        if up_to_round is not None and r > up_to_round:
            break

        rx_df = pd.DataFrame.from_records(rx_events)
        tx_df = pd.DataFrame.from_records(tx_events)

        for transmitter in range(len(testbed.devs)):
            for receiver in range(len(testbed.devs)):
                if receiver != transmitter:
                    if len(rx_df.index) > 0 and len(tx_df.index) > 0:
                        pairs = (extract_rx_tx_pairs(rx_df, tx_df, transmitter, receiver))
                        rx_var_est = estimate_noise_std_with_lls_grouped_median(pairs)

                        #pairs = (extract_rx_tx_pairs(rx_df, tx_df, transmitter, receiver))
                        #rx_var_est_cfo = estimate_noise_std_with_cfo_mean(pairs)

                        e = {
                            'round': r,
                            'tx_number': transmitter,
                            'rx_number': receiver,
                            'rx_std_est': rx_var_est
                        }

                        rows_list.append(
                           e
                        )

    df = pd.DataFrame(rows_list)
    return df


def extract_data_slots(rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot, final_slot):
    # we extract data for initiator a and responder b (numbers not device names!)

    def first_or_none(l):
        return next(iter(l), None)

    init_tx = first_or_none(
        tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == init_slot) & (tx_df["own_number"] == a)].to_dict(
            orient='records'))
    init_rx = first_or_none(
        rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == init_slot) & (rx_df["own_number"] == b)].to_dict(
            orient='records'))
    init_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == init_slot) & (
                rx_df["own_number"] == tdoa_src_dev_number)].to_dict(orient='records'))

    response_tx = first_or_none(
        tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == response_slot) & (tx_df["own_number"] == b)].to_dict(
            orient='records'))
    response_rx = first_or_none(
        rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == response_slot) & (rx_df["own_number"] == a)].to_dict(
            orient='records'))
    response_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == response_slot) & (
                rx_df["own_number"] == tdoa_src_dev_number)].to_dict(orient='records'))

    final_tx = first_or_none(
        tx_df[(tx_df["tx_round"] == r) & (tx_df["tx_slot"] == final_slot) & (tx_df["own_number"] == a)].to_dict(
            orient='records'))
    final_rx = first_or_none(
        rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == final_slot) & (rx_df["own_number"] == b)].to_dict(
            orient='records'))
    final_rx_passive = first_or_none(rx_df[(rx_df["rx_round"] == r) & (rx_df["rx_slot"] == final_slot) & (
                rx_df["own_number"] == tdoa_src_dev_number)].to_dict(orient='records'))

    assert init_rx is None or init_rx['rx_number'] == a
    assert init_rx_passive is None or init_rx_passive['rx_number'] == a

    assert response_rx is None or response_rx['rx_number'] == b
    assert response_rx_passive is None or response_rx_passive['rx_number'] == b

    assert final_rx is None or final_rx['rx_number'] == a
    assert final_rx_passive is None or final_rx_passive['rx_number'] == a

    ret = {
        'init_tx': init_tx,
        'init_rx': init_rx,
        'init_rx_passive': init_rx_passive,
        'response_tx': response_tx,
        'response_rx': response_rx,
        'response_rx_passive': response_rx_passive,
        'final_tx': final_tx,
        'final_rx': final_rx,
        'final_rx_passive': final_rx_passive
    }

    return ret




def extract_record(testbed, rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot, final_slot, bias_corrected=True):

    data = extract_data_slots(rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot, final_slot)

    da = testbed.devs[a]
    db = testbed.devs[b]

    if tdoa_src_dev_number is not None:
        tdoa_src_dev = testbed.devs[tdoa_src_dev_number]
    else:
        tdoa_src_dev = None


    record = {}

    record['round'] = int(r)
    record['initiator'] = a
    record['responder'] = b
    record['pair'] = "{}-{}".format(a, b)
    record['dist'] = get_dist(testbed.dev_positions[da], testbed.dev_positions[db])
    record['tdoa'] = None

    # we check if data contains any None values, if so, we drop this exchange
    def dur(start, end):
        if None in [start, end]:
            return None

        if end <= start:
            end += 0xFFFFFFFFFF + 1  # we handle overflow here

        return end - start

    round_a = dur((data.get('init_tx', {}) or {}).get('tx_ts', None),
                  (data.get('response_rx', {}) or {}).get(
                      'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None))
    delay_b = dur(
        (data.get('init_rx', {}) or {}).get('bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None),
        (data.get('response_tx', {}) or {}).get('tx_ts', None))
    delay_a = dur(
        (data.get('response_rx', {}) or {}).get('bias_corrected_rx_ts' if bias_corrected else 'rx_ts',
                                                None), (data.get('final_tx', {}) or {}).get('tx_ts', None))
    round_b = dur((data.get('response_tx', {}) or {}).get('tx_ts', None),
                  (data.get('final_rx', {}) or {}).get(
                      'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None))

    record['round_a'] = round_a
    record['delay_b'] = delay_b
    record['delay_a'] = delay_a
    record['round_b'] = round_b

    record['init_rx_phase'] = (data.get('init_rx', {}) or {}).get('rx_ttcko_rc_phase', None)
    record['init_rx_passive_phase'] = (data.get('init_rx_passive', {}) or {}).get('rx_ttcko_rc_phase', None)
    record['response_rx_phase'] = (data.get('response_rx', {}) or {}).get('rx_ttcko_rc_phase', None)
    record['response_rx_passive_phase'] = (data.get('response_passive_rx', {}) or {}).get('rx_ttcko_rc_phase', None)
    record['final_rx_phase'] = (data.get('final_rx', {}) or {}).get('rx_ttcko_rc_phase', None)
    record['final_rx_passive_phase'] = (data.get('final_rx_passive', {}) or {}).get('rx_ttcko_rc_phase', None)

    def ci_or_none_to_rd(ci):
        if ci is None:
            return None
        else:
            return ci_to_rd(ci)

    record['init_rd_cfo'] = ci_or_none_to_rd((data.get('init_rx', {}) or {}).get('ci', None))
    record['response_rd_cfo'] = ci_or_none_to_rd((data.get('response_rx', {}) or {}).get('ci', None))
    record['final_rd_cfo'] = ci_or_none_to_rd((data.get('final_rx', {}) or {}).get('ci', None))

    if None not in [round_a, delay_b, delay_a, round_b]:
        relative_drift = float(round_a + delay_a) / float(round_b + delay_b)
        twr_tof = convert_ts_to_m((round_a - relative_drift * delay_b) * 0.5)

        record['relative_drift'] = relative_drift
        record['twr_tof_ds'] = twr_tof

        record['twr_tof_ss'] = convert_ts_to_m((round_a - record['response_rd_cfo'] * delay_b) * 0.5)
        record['twr_tof_ss_reverse'] = convert_ts_to_m((round_b - record['final_rd_cfo'] * delay_a) * 0.5)

        if tdoa_src_dev:

            record['tdoa'] = get_dist(testbed.dev_positions[da],
                                      testbed.dev_positions[tdoa_src_dev]) - get_dist(
                testbed.dev_positions[tdoa_src_dev], testbed.dev_positions[db])
            record['tdoa_device'] = tdoa_src_dev_number

            passive_m_a = dur((data.get('init_rx_passive', {}) or {}).get(
                'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None),
                (data.get('response_rx_passive', {}) or {}).get(
                    'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None))
            passive_m_b = dur((data.get('response_rx_passive', {}) or {}).get(
                'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None),
                (data.get('final_rx_passive', {}) or {}).get(
                    'bias_corrected_rx_ts' if bias_corrected else 'rx_ts', None))

            if None not in [passive_m_a, passive_m_b]:
                record['passive_m_a'] = passive_m_a
                record['passive_m_b'] = passive_m_b

                record['tdoa_a_relative_drift_ds'] = (record['passive_m_a'] + record['passive_m_b']) / (
                        record['round_a'] + record['delay_a'])
                record['tdoa_b_relative_drift_ds'] = (record['passive_m_a'] + record['passive_m_b']) / (
                        record['round_b'] + record['delay_b'])

                record['tdoa_init_rd_cfo'] = ci_to_rd((data.get('init_rx_passive', {}) or {}).get('ci'))
                record['tdoa_response_rd_cfo'] = ci_to_rd(
                    (data.get('response_rx_passive', {}) or {}).get('ci'))
                record['tdoa_final_rd_cfo'] = ci_to_rd((data.get('final_rx_passive', {}) or {}).get('ci'))

                record['tdoa_est_ds'] = convert_ts_to_m(
                    0.5 * record['tdoa_a_relative_drift_ds'] * round_a + 0.5 * record[
                        'tdoa_b_relative_drift_ds'] * delay_b - passive_m_a)
                record['tdoa_est_ss_init'] = convert_ts_to_m(
                    0.5 * record['tdoa_init_rd_cfo'] * round_a + 0.5 * record[
                        'tdoa_response_rd_cfo'] * delay_b - passive_m_a)
                record['tdoa_est_ss_final'] = convert_ts_to_m(
                    0.5 * record['tdoa_final_rd_cfo'] * round_a + 0.5 * record[
                        'tdoa_response_rd_cfo'] * delay_b - passive_m_a)
                record['tdoa_est_ss_both'] = convert_ts_to_m(
                    0.25 * record['tdoa_init_rd_cfo'] * round_a + 0.25 * record[
                        'tdoa_final_rd_cfo'] * round_a + 0.5 * record[
                        'tdoa_response_rd_cfo'] * delay_b - passive_m_a)
                record['tdoa_est_mixed'] = convert_ts_to_m(
                    0.5 * record['tdoa_a_relative_drift_ds'] * round_a + 0.5 * record[
                        'tdoa_response_rd_cfo'] * delay_b - passive_m_a)
    return record


def gen_tdma_twr_resp_delays_records(testbed, run, tdoa_src_dev_number=None, bias_corrected=True):

    for (r, rx_events, tx_events) in gen_round_events(testbed, run):
        rx_df = pd.DataFrame.from_records(rx_events)
        tx_df = pd.DataFrame.from_records(tx_events)

        print(r, len(rx_events), len(tx_events))

        if len(rx_events) == 0 or len(tx_events) == 0:
            continue

        exchanges = range(32)
        init_devs = testbed.devs[:1]
        resp_devs = testbed.devs[1:2]

        for exchange in exchanges:
            for (a, da) in enumerate(init_devs):
                for (b, db) in enumerate(resp_devs):

                    if a == b:
                        continue

                    # we extract data for initiator a and responder b (numbers not device names!)
                    init_slot = exchange * 3
                    response_slot = init_slot + 1
                    final_slot = response_slot + 1
                    rec = extract_record(testbed, rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot, final_slot, bias_corrected=bias_corrected)
                    #print(rec)
                    yield rec

def gen_tdma_twr_records(testbed, run, tdoa_src_dev_number=None, bias_corrected=True, experiment='RESP_DELAYS'):

    for (r, rx_events, tx_events) in gen_round_events(testbed, run):
        rx_df = pd.DataFrame.from_records(rx_events)
        tx_df = pd.DataFrame.from_records(tx_events)

        print(r, len(rx_events), len(tx_events))

        if len(rx_events) == 0 or len(tx_events) == 0:
            continue


        for (a, da) in enumerate(testbed.devs):
            for (b, db) in enumerate(testbed.devs):

                if a == b:
                    continue

                # we extract data for initiator a and responder b (numbers not device names!)
                init_slot = a * (len(testbed.devs) - 1) * 3 + b * 3

                if a <= b:
                    init_slot -= 3  # we save one exchange because a does not exchange it with itself...

                response_slot = init_slot + 1
                final_slot = response_slot + 1

                yield extract_record(testbed, rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot, final_slot, bias_corrected=bias_corrected)


def gen_ping_pong_records(testbed, run, tdoa_src_dev_number=None, bias_corrected=True, max_slot_dur=None):

    ping_pong_initiator_number = 3
    ping_pong_slots = 200+1

    if max_slot_dur is None:
        max_slot_dur = ping_pong_slots - 1

    assert max_slot_dur % 2 == 0

    #ping_pong_initiator = testbed.devs[ping_pong_initiator_number]

    for (r, rx_events, tx_events) in gen_round_events(testbed, run):
        rx_df = pd.DataFrame.from_records(rx_events)
        tx_df = pd.DataFrame.from_records(tx_events)

        print(r, len(rx_events), len(tx_events))

        if len(rx_events) == 0 or len(tx_events) == 0:
            continue

        for (b, db) in enumerate(testbed.devs):

            if ping_pong_initiator_number == b:
                continue

            multiplier = b if b < ping_pong_initiator_number else b-1

            start_slot = ping_pong_slots * multiplier
            end_slot = start_slot + ping_pong_slots - 1

            if start_slot % 2 == 1:
                # due to a wrong modulo calculation, we have to correct the start_slot in this case!
                # we simply reduce the start and end_slot by 1
                start_slot -= 1
                end_slot -= 1

            for init_slot in range(start_slot, end_slot, max_slot_dur):
                final_slot = init_slot + max_slot_dur

                if final_slot > end_slot:
                    break

                for response_slot in range(init_slot+1, final_slot, 2):
                    #print(b, multiplier, init_slot, response_slot, final_slot)
                    rec = extract_record(testbed, rx_df, tx_df, r, ping_pong_initiator_number, b, tdoa_src_dev_number, init_slot, response_slot, final_slot, bias_corrected=bias_corrected)
                    #print(rec['relative_drift'])
                    yield rec


def gen_new_delay_records(testbed, run, tdoa_src_dev_number=None, bias_corrected=True, initiator_id=None, responder_id=None, num_exchanges=100):
    slot_offset = 1 # we have one initiator message at the beginning
    for (r, rx_events, tx_events) in gen_round_events(testbed, run):
        rx_df = pd.DataFrame.from_records(rx_events)
        tx_df = pd.DataFrame.from_records(tx_events)

        print(r, len(rx_events), len(tx_events))

        if len(rx_events) == 0 or len(tx_events) == 0:
            continue

        exchanges = range(num_exchanges)

        for exchange in exchanges:
            a = initiator_id
            b = responder_id


            # we extract data for initiator a and responder b (numbers not device names!)
            init_slot = exchange * 3 + slot_offset
            response_slot = init_slot + 1
            final_slot = response_slot + 1
            rec = extract_record(testbed, rx_df, tx_df, r, a, b, tdoa_src_dev_number, init_slot, response_slot,
                                 final_slot, bias_corrected=bias_corrected)
            yield rec

def extract_tdma_twr(testbed, run, tdoa_src_dev_number=None, bias_corrected=True, experiment='RESP_DELAYS'):
    if experiment == 'RESP_DELAYS':
        yield from gen_tdma_twr_resp_delays_records(testbed, run, tdoa_src_dev_number, bias_corrected)
    else:
        yield from gen_tdma_twr_records(testbed, run, tdoa_src_dev_number, bias_corrected)

# the combined phase should be pa + pb = (4* pi * dist) * (freq / c) + 4 pi N
# we choose N such that the distance is close to the measured one

# (pa + pb) - 4 pi N = (4* pi * dist) * (freq / c)
# ((pa + pb) - 4 pi N) / (4 * pi)  = dist * (freq / c)
# (((pa + pb) - 4 pi N) / (4 * pi)) * (c / freq) = dist
# (((pa + pb)/ (4 * pi) - (4 pi N) / (4 * pi)) * (c / freq) = dist
# (((pa + pb)/ (4 * pi) - N) * (c / freq) = dist

# we compute 3 N and select the one that fits the best, N0 := floor(dist / (c / freq))-1, N1 := N0+1, N2:=N0+2
import base
def compute_phase_dist(comb_phase, measured_dist):
    comb_phase_in_radians = (comb_phase/128.0) * (2*np.pi)
    c = base.SPEED_OF_LIGHT
    freq = 6.5 * 1000000000

    N0 = np.floor(measured_dist / (c / freq)) - 1
    N1 = N0 + 1
    N2 = N0 + 2
    N3 = N0 + 3

    d0 = (comb_phase_in_radians / (4 * np.pi) + N0) * (c / freq)
    d1 = (comb_phase_in_radians / (4 * np.pi) + N1) * (c / freq)
    d2 = (comb_phase_in_radians / (4 * np.pi) + N2) * (c / freq)
    d3 = (comb_phase_in_radians / (4 * np.pi) + N3) * (c / freq)


    diff0 = np.abs(d0 - measured_dist)
    diff1 = np.abs(d1 - measured_dist)
    diff2 = np.abs(d2 - measured_dist)
    diff3 = np.abs(d2 - measured_dist)

    m = np.min([diff0, diff1, diff2, diff3])
    assert m is None or np.isnan(m) or m <= 0.047
    assert diff0 is None or np.isnan(diff0) or d0 <= measured_dist
    assert d3 is None or np.isnan(d3) or d3 >= measured_dist

    if m == diff0:
        return d0
    elif m == diff1:
        return d1
    elif m == diff2:
        return d2
    elif m == diff3:
        return d3
    else:
        return None


def apply_phase_dist_to_col(d):
    xs = []
    for (comb_phases, meas_dist) in zip(d['combined_phase'].to_numpy(), d['twr_tof_ds'].to_numpy()):
        xs.append(compute_phase_dist(comb_phases, meas_dist))
    return np.asarray(xs)



if __name__ == '__main__':
    import testbed.trento_a as trento_a

    it  = gen_ping_pong_records(trento_a, 'ping_pong_trento_a_source_4_11702', tdoa_src_dev_number=None, bias_corrected=True)
    df = pd.DataFrame.from_records(it)
    print(df)



#     import testbed.trento_b as trento_b
#     from utility import cached_dt
#     from utility import init_cache, load_env_config, set_global_cache_prefix_by_config
#
#     config = load_env_config()
#
#     assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
#
#     if 'CACHE_DIR' in config and config['CACHE_DIR']:
#         init_cache(config['CACHE_DIR'])
#
#     skip_to_round = 25
#     up_to_round = 99999999
#     use_bias_correction = True
#     tdoa_src_dev_number = 0
#
#     log = 'job_11063' # path resolves to data/trento_b/exp_rx_noise_10041.log
#
#     def extract():
#         it = extract_tdma_twr(trento_b, log, tdoa_src_dev_number=tdoa_src_dev_number,
#                               bias_corrected=use_bias_correction, experiment='TWR')
#         return pd.DataFrame.from_records(it)
#
#
#     df = cached_dt(('extract_job_tdma_new_4', log, tdoa_src_dev_number, use_bias_correction), extract)
#
#
#     # the actual response rx phase is skewed by our local drift relative to b
#     # hence we have to adjust the rx phase accordingly
#     # if k_A > k_B, our phase drifts further while executing the protocol,
#     # our own clock would be approximately (k_A/k_B)*Delay_A further
#     # one phase duration is approx 0.046m / c = 0.000000153333333s
#     # speaking of phase measurements, this would be ((k_A/k_B)*Delay_A) / ((c / freq) / c)
#     # so ((k_A/k_B)*Delay_A) * freq
#     #
#     # note that we could also adapt the inital_rx phase and correct accordingly...
#     # we hence have to account for that as well
#     # for now, we reuse the extracted relative drift from the DS-TWR..
#     # note that we are NOT wrapping around! as we might have missed multiple phases...
#     #
#     resp_corr = convert_ts_to_sec(df['round_a']) * base.CHAN5_FREQ
#     # TODO: We might need to put not the rx timestamp into this calculation...
#
#     resp_corr = resp_corr-np.floor(resp_corr)
#     df['response_rx_phase_corrected'] = np.mod((df['response_rx_phase'] - resp_corr * 128.0 + 128.0), 128.0)
#
#     #df['combined_phase'] = (df['init_rx_phase'] - df['response_rx_phase'])
#     df['combined_phase'] = df['init_rx_phase'] - df['response_rx_phase_corrected']
#     df['phase_dist'] = apply_phase_dist_to_col(df)
#
#
#     df = df[(df['round'] >= skip_to_round) & (df['round'] <= up_to_round)]
#
#     df['phase_dist_err'] = df['phase_dist'] - df['dist']
#     df['twr_tof_ds_err'] = df['twr_tof_ds'] - df['dist']
#     df['twr_tof_ss_err'] = df['twr_tof_ss'] - df['dist']
#     df['twr_tof_ss_reverse_err'] = df['twr_tof_ss_reverse'] - df['dist']
#     df['tdoa_est_ds_err'] = df['tdoa_est_ds'] - df['tdoa']
#     df['tdoa_est_ss_init_err'] = df['tdoa_est_ss_init'] - df['tdoa']
#     df['tdoa_est_ss_final_err'] = df['tdoa_est_ss_final'] - df['tdoa']
#     df['tdoa_est_ss_both_err'] = df['tdoa_est_ss_both'] - df['tdoa']
#     df['tdoa_est_mixed_err'] = df['tdoa_est_mixed'] - df['tdoa']
#
#
#
#     phase_dist_err = df['phase_dist_err'].to_numpy()
#     twr_tof_ds_err = df['twr_tof_ds_err'].to_numpy()
#
#     print("Phase: {}, DS-TWR: {}".format(np.nanstd(phase_dist_err), np.nanstd(twr_tof_ds_err)))
#     exit()
#
#
#     #pair_df = df[df['pair'] == '0-1']
#     pair_df = df
#
#     #pair_df['init_rx_phase_5deg'] = np.round(pair_df['init_rx_phase'] / 5)
#     #pair_df['response_rx_phase_5deg'] = np.round(pair_df['response_rx_phase'] / 5)
#
#
#
#
#
#     pair_df = pair_df.groupby('combined_phase').agg(
#         count=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="count"),
#         phase=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="count"),
#         twr_tof_ds_err_mean=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="mean"),
#         twr_tof_ds_err_std=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="std"),
#     )
#
#     import matplotlib
#
#     import matplotlib.pyplot as plt
#
#
# #    df.to_csv('raw-logs-{}-out-{}.csv'.format(log, tdoa_src_dev_number))
#
#     gb = df.groupby('pair').agg(
#         count=pd.NamedAgg(column='tdoa_est_ds', aggfunc="count"),
#         dist=pd.NamedAgg(column='dist', aggfunc="min"),
#         tdoa=pd.NamedAgg(column='tdoa', aggfunc="min"),
#         phase_dist_err_mean=pd.NamedAgg(column='phase_dist_err', aggfunc="mean"),
#         phase_dist_err_std=pd.NamedAgg(column='phase_dist_err', aggfunc="std"),
#         twr_tof_ds_err_mean=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="mean"),
#         twr_tof_ds_err_std=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="std"),
#         twr_tof_ss_err_mean=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="mean"),
#         twr_tof_ss_err_std=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="std"),
#         twr_tof_ss_reverse_err_mean=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="mean"),
#         twr_tof_ss_reverse_err_std=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="std"),
#         tdoa_est_ds_err_mean=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="mean"),
#         tdoa_est_ds_err_std=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="std"),
#         tdoa_est_ss_init_err_mean=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="mean"),
#         tdoa_est_ss_init_err_std=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="std"),
#         tdoa_est_ss_both_err_mean=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="mean"),
#         tdoa_est_ss_both_err_std=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="std"),
#         tdoa_est_ss_final_err_mean=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="mean"),
#         tdoa_est_ss_final_err_std=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="std"),
#         tdoa_est_mixed_err_mean=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="mean"),
#         tdoa_est_mixed_err_std=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="std")
#     )
#
#     gb.to_csv('raw-logs-{}-out-pairs-{}.csv'.format(log, tdoa_src_dev_number))
#
#     gb.plot.bar(y=['twr_tof_ds_err_std', 'phase_dist_err_std'])
#
#     plt.show()



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


# 1 & XX & 10.34 & XX \\  \hline