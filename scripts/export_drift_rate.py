import os
import progressbar
import numpy as np
import json

import scipy.optimize

import logs
import utility

from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, \
    gen_delay_estimates_from_testbed_run
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd

import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached_legacy, init_cache, load_env_config, set_global_cache_prefix_by_config
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from export import add_df_cols, load_plot_defaults, save_and_crop, c_in_air, CONFIDENCE_FILL_COLOR, PERCENTILES_FILL_COLOR, COLOR_MAP, PROTOCOL_NAME
import pandas as pd


from export_ping_pong_std import prepare_df, get_df


logfiles = [
        '2024-02-28_ping_pong_200/job_11985.tar.gz',
        '2024-02-28_ping_pong_200/job_11986.tar.gz',
        '2024-02-28_ping_pong_200/job_11987.tar.gz',
        '2024-02-28_ping_pong_200/job_11988.tar.gz',
        # '2024-02-28_ping_pong_200/job_11989.tar.gz',
        # '2024-02-28_ping_pong_200/job_11990.tar.gz',
        # '2024-02-28_ping_pong_200/job_11991.tar.gz',
        # '2024-02-28_ping_pong_200/job_11992.tar.gz',
        # '2024-02-28_ping_pong_200/job_11993.tar.gz',
        # '2024-02-28_ping_pong_200/job_11994.tar.gz',
        # '2024-02-28_ping_pong_200/job_11995.tar.gz',
        # '2024-02-28_ping_pong_200/job_11996.tar.gz',
        # '2024-02-28_ping_pong_200/job_11997.tar.gz',
        # '2024-02-28_ping_pong_200/job_11998.tar.gz',
        # '2024-02-28_ping_pong_200/job_11999.tar.gz',
        # '2024-03-01_ping_pong_200/job_12004.tar.gz',
        # '2024-03-01_ping_pong_200/job_12005.tar.gz',
        # '2024-03-01_ping_pong_200/job_12006.tar.gz',
        # '2024-03-01_ping_pong_200/job_12007.tar.gz',
        # '2024-03-01_ping_pong_200/job_12008.tar.gz',
        # '2024-03-01_ping_pong_200/job_12009.tar.gz',
        # '2024-03-01_ping_pong_200/job_12010.tar.gz',
        # '2024-03-01_ping_pong_200/job_12011.tar.gz',
        # '2024-03-01_ping_pong_200/job_12012.tar.gz',
]

passive_dev = 6
max_slot_dur = 22

CHOSEN_DUR=82

def get_noise_df(log, max_slots_dur, use_bias_correction=True):
    def proc():
        print("Processing", log, max_slots_dur)
        it = logs.gen_ping_pong_rx_noise_records(trento_a, log, bias_corrected=use_bias_correction,
                                            max_slot_dur=max_slots_dur)
        df = pd.DataFrame.from_records(it)
        return df

    df = utility.cached_dt_legacy(('get_noise_df_12', log, use_bias_correction, max_slots_dur), proc)
    df['dur_ms'] = max_slots_dur * 0.75

    return df


def est_noise_std(rss, n):
    return np.sqrt(rss / (n - 2))

def est_noise_std_ci(rss, num, alpha=0.05):
    lower_bound = rss / scipy.stats.chi2.ppf(1-alpha/2.0, num-2)
    upper_bound = rss / scipy.stats.chi2.ppf(alpha/2.0, num-2)

    return np.sqrt(lower_bound), np.sqrt(upper_bound)


def estimate_reception_noise_map(max_slots_dur=None, use_bias_correction=True, min_round = 50):

    if max_slots_dur is None:
        max_slots_dur = CHOSEN_DUR
    dfs = [
        get_noise_df(log, max_slots_dur=max_slots_dur) for log in logfiles
    ]
    df = pd.concat(dfs, ignore_index=True, copy=True)

    rx_noise_map = {}
    for initiator in [0, 1, 2, 3, 4, 5, 6]:
        for responder in [0, 1, 2, 3, 4, 5, 6]:
            if initiator == responder:
                continue

            fdf = df[(df['initiator'] == initiator) & (df['responder'] == responder)]

            if min_round is not None:
                fdf = fdf[fdf['round'] >= min_round]

            # df_aggr = fdf.agg(
            #     rx_std_est_q25=('rx_std_est', lambda x: x.quantile(0.05)),
            #     rx_std_est_q50=('rx_std_est', lambda x: x.quantile(0.50)),
            #     rx_std_est_q75=('rx_std_est', lambda x: x.quantile(0.95)),
            #     rx_std_est_min=('rx_std_est', 'min'),
            #     rx_std_est_max=('rx_std_est', 'max'),
            #     rx_std_est_mean=('rx_std_est', 'mean')
            # )

            est_rx_ssr = fdf['est_rx_ssr'].to_numpy(dtype='float')
            est_rx_num = fdf['est_rx_num'].to_numpy(dtype='float')

            sd = convert_ts_to_m(est_noise_std(est_rx_ssr, est_rx_num))

            # ci = (est_noise_std_ci(est_rx_ssr, est_rx_num))
            # #print(sd, ci)
            # print(max_slots_dur, np.median(sd), convert_ts_to_m(np.median(ci[0])), convert_ts_to_m(np.median(ci[1])))
            # exit()

            rx_noise_map[(initiator, responder)] = np.median(sd)

    return rx_noise_map


def export_drift(export_dir):

    min_round = 50
    for initiator in [0, 1, 2, 3, 4, 5, 6]:
        for responder in [0, 1, 2, 3, 4, 5, 6]:
            if initiator == responder:
                continue

            dfs = [
                get_noise_df(log, max_slots_dur=dur) for log in logfiles for dur in list(range(10, 200+1, 4))
            ]

            df = pd.concat(dfs, ignore_index=True, copy=True)

            fdf = df[(df['initiator'] == initiator) & (df['responder'] == responder)]

            if min_round is not None:
                fdf = fdf[fdf['round'] >= min_round]

            est_rx_ssr = fdf['est_rx_ssr'].to_numpy(dtype='double')
            est_rx_num = fdf['est_rx_num'].to_numpy(dtype='double')

            fdf['est_rx_sd'] = convert_ts_to_m(est_noise_std(est_rx_ssr, est_rx_num))
            ci = (est_noise_std_ci(est_rx_ssr, est_rx_num))
            fdf['est_rx_ci_low'] = convert_ts_to_m(ci[0])
            fdf['est_rx_ci_high'] = convert_ts_to_m(ci[1])

            df_aggr = fdf.groupby('dur_ms').agg(
                dur_ms=('dur_ms', 'mean'),
                est_rx_sd=('est_rx_sd', 'median'),
                est_rx_ci_low=('est_rx_ci_low', 'median'),
                est_rx_ci_high=('est_rx_ci_high', 'median'),
                count=('dur_ms', 'count')
            )

            # ci = (est_noise_std_ci(est_rx_ssr, est_rx_num))
            # #print(sd, ci)
            # print(max_slots_dur, np.median(sd), convert_ts_to_m(np.median(ci[0])), convert_ts_to_m(np.median(ci[1])))
            # exit()
            #
            # est_rx_sample_variance = np.asarray(df_aggr['est_rx_ssr'].to_numpy(dtype='double') / (df_aggr['est_rx_num'].to_numpy(dtype='double') - 2.0))
            # est_rx_ci_low, est_rx_ci_up = logs.calc_ci_of_sd(np.sqrt(est_rx_sample_variance), df_aggr['est_rx_num'].to_numpy(dtype='double'))
            # est_rx_ci_low, est_rx_ci_up = (convert_ts_to_m(est_rx_ci_low), convert_ts_to_m(est_rx_ci_up))
            # est_rx_sample_variance = convert_ts_to_m(np.sqrt(est_rx_sample_variance))

            # est_cfo_median_sample_variance = df_aggr['est_cfo_median_ssr'].to_numpy(dtype='double') / (df_aggr['est_cfo_median_num'].to_numpy(dtype='double') - 2)
            # est_cfo_median_ci_low, est_cfo_median_ci_up = logs.calc_ci_of_sd(np.sqrt(est_cfo_median_sample_variance),
            #                                                  df_aggr['est_cfo_median_num'].to_numpy(dtype='double'))
            # est_cfo_median_ci_low, est_cfo_median_ci_up = (convert_ts_to_m(est_cfo_median_ci_low), convert_ts_to_m(est_cfo_median_ci_up))
            # est_cfo_median_sample_variance = convert_ts_to_m(np.sqrt(est_cfo_median_sample_variance))
            #
            # est_cfo_mean_sample_variance = df_aggr['est_cfo_mean_ssr'].to_numpy(dtype='double') / (
            #             df_aggr['est_cfo_mean_num'].to_numpy(dtype='double') - 2)
            # est_cfo_mean_ci_low, est_cfo_mean_ci_up = logs.calc_ci_of_sd(np.sqrt(est_cfo_mean_sample_variance),
            #                                                                  df_aggr['est_cfo_mean_num'].to_numpy(dtype='double'))
            # est_cfo_mean_ci_low, est_cfo_mean_ci_up = (convert_ts_to_m(est_cfo_mean_ci_low), convert_ts_to_m(est_cfo_mean_ci_up))
            # est_cfo_mean_sample_variance = convert_ts_to_m(np.sqrt(est_cfo_mean_sample_variance))


            #df_aggr['est_rx_sd'] = est_rx_sample_variance
            #df_aggr['est_cfo_mean_sd'] = est_cfo_mean_sample_variance
            #df_aggr['est_cfo_median_sd'] = est_cfo_median_sample_variance

            fig, ax = plt.subplots()
            #df_aggr.plot.line(y='num', ax=ax, label="num")
            df_aggr.plot.line(y='est_rx_sd', ax=ax, label="est_rx_sd")
            #df_aggr.plot.line(y='est_cfo_mean_sd', ax=ax, label="est_cfo_mean_sd")
            #df_aggr.plot.line(y='est_cfo_median_sd', ax=ax, label="est_cfo_median_sd")
            #df_aggr.plot.line(y='rx_std_est_q50', ax=ax, label="Median")
            #df_aggr.plot.line(y='rx_std_est_mean', ax=ax, label="Mean")
            plt.fill_between(df_aggr['dur_ms'], df_aggr['est_rx_ci_low'], df_aggr['est_rx_ci_high'], alpha=0.25)
            #plt.fill_between(df_aggr['dur_ms'], , ci_up, alpha=0.25)

            plt.axvline(CHOSEN_DUR*0.75, color='r', linestyle='--', label="Chosen Duration ({} ms)".format(round(CHOSEN_DUR*0.75)))

            ax.set_ylim(0, 0.05)

            save_and_crop("{}/drift_rate_noise_{}_{}_min_round_{}.pdf".format(export_dir, initiator, responder, min_round), bbox_inches='tight')  # , pad_inches=0)
            #plt.show()
            plt.close()

def export_drift_over_rounds(export_dir):
    for initiator in [3]:
        for responder in [0, 1, 2, 4, 5, 6]:
            for dur in [14]: # duration does not really affect this graph, higher values just smoothen the graph
                dfs = [
                    get_noise_df(log, max_slots_dur=dur) for log in logfiles
                ]

                df = pd.concat(dfs, ignore_index=True, copy=True)

                df['log_round_start_ts'] = pd.to_datetime(df['log_round_start_ts'], infer_datetime_format=True, errors='ignore').map(
                    pd.Timestamp.timestamp)

                df['log_round_start_s'] = df['log_round_start_ts'].round() - df['log_round_start_ts'].min()

                fdf = df[(df['initiator'] == initiator) & (df['responder'] == responder)]

                df_aggr = fdf.groupby('log_round_start_s').agg(
                    est_rx_drift_mean_q50=('est_rx_ref_rd', lambda x: x.quantile(0.50)),
                    est_cfo_mean_drift_mean_q50=('est_cfo_mean_ref_rd', lambda x: x.quantile(0.50)),
                    est_cfo_median_drift_mean_q50=('est_cfo_median_ref_rd', lambda x: x.quantile(0.50)),
                    log_round_start_s=('log_round_start_s', 'mean')
                )

                fig, ax = plt.subplots()
                df_aggr.plot.line(y='est_rx_drift_mean_q50', ax=ax, label="RX est_rx_drift_mean_q50")
                df_aggr.plot.line(y='est_cfo_mean_drift_mean_q50', ax=ax, label="RX est_cfo_mean_drift_mean_q50")
                df_aggr.plot.line(y='est_cfo_median_drift_mean_q50', ax=ax, label="RX est_cfo_median_drift_mean_q50")
                #plt.fill_between(df_aggr['log_round_start_s'], df_aggr['drift_mean_q25'], df_aggr['drift_mean_q75'], alpha=0.25)

                save_and_crop("{}/drift_rate_per_round_{}_{}_dur_{}.pdf".format(export_dir, initiator, responder, dur), bbox_inches='tight')  # , pad_inches=0)
                plt.close()


def export_drift_rate_over_rounds(export_dir):
    for initiator in [3]:
        for responder in [0, 1, 2, 4, 5, 6]:
            for dur in range(14, 201, 20):
                dfs = [
                    get_noise_df(log, max_slots_dur=dur) for log in logfiles
                ]

                df = pd.concat(dfs, ignore_index=True, copy=True)

                df['log_round_start_ts'] = pd.to_datetime(df['log_round_start_ts'], infer_datetime_format=True).map(
                    pd.Timestamp.timestamp)

                df['log_round_start_s'] = df['log_round_start_ts'].round() - df['log_round_start_ts'].min()

                fdf = df[(df['initiator'] == initiator) & (df['responder'] == responder)]

                df_aggr = fdf.groupby('log_round_start_s').agg(
                    rx_std_est_q25=('rx_std_est', lambda x: x.quantile(0.25)),
                    rx_std_est_q50=('rx_std_est', lambda x: x.quantile(0.50)),
                    rx_std_est_q75=('rx_std_est', lambda x: x.quantile(0.75)),
                    log_round_start_s=('log_round_start_s', 'mean'),
                )

                fig, ax = plt.subplots()
                df_aggr.plot.line(y='rx_std_est_q50', ax=ax, label="RX Est")
                plt.fill_between(df_aggr['log_round_start_s'], df_aggr['rx_std_est_q25'], df_aggr['rx_std_est_q75'], alpha=0.25)

                save_and_crop("{}/drift_noise_per_round_{}_{}_dur_{}.pdf".format(export_dir, initiator, responder, dur), bbox_inches='tight')  # , pad_inches=0)
                plt.close()
if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    rx_noise_map = estimate_reception_noise_map(CHOSEN_DUR)
    print([rx_noise_map.get((3, i), None) for i in range(7)])

    export_drift(config['EXPORT_DIR'])
    #export_drift_over_rounds(config['EXPORT_DIR'])
    #export_drift_rate_over_rounds(config['EXPORT_DIR'])




