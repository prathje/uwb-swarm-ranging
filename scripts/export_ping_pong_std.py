import os
import progressbar
import numpy as np
import json

import scipy.optimize

import logs
import utility
from eval_old import calc_predicted_tof_std_navratil


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

use_bias_correction = True
import csv



def get_df(log, tdoa_src_dev_number, max_slots_dur):
    def proc():
        print("Processing", log, tdoa_src_dev_number, max_slots_dur)
        it = logs.gen_ping_pong_records(trento_a, log, tdoa_src_dev_number=tdoa_src_dev_number,
                                        bias_corrected=use_bias_correction, max_slot_dur=max_slots_dur)
        df = pd.DataFrame.from_records(it)
        return add_df_cols(df, tdoa_src_dev_number)


    return utility.cached_dt_legacy(  # todo: this was 3
        ('extract_job_tdma_ping_pong_4', log, tdoa_src_dev_number, use_bias_correction, max_slots_dur), proc)



def prepare_df(df, initiator=3, responder=5, min_round=50):

    df = df[df['initiator'] == initiator]
    df = df[df['responder'] == responder]

    df['twr_tof_ss_avg'] = (df['twr_tof_ss'] + df['twr_tof_ss_reverse']) / 2

    df['delay_b_ms'] = df['delay_b'].apply(lambda x: convert_ts_to_sec(x) * 1000.0)
    df['delay_a_ms'] = df['delay_a'].apply(lambda x: convert_ts_to_sec(x) * 1000.0)


    df['delay_b_ms_rounded'] = df['delay_b_ms'].apply(lambda x: np.round(x, decimals=1))
    df['delay_a_ms_rounded'] = df['delay_a_ms'].apply(lambda x: np.round(x, decimals=1))

    df = df[df['delay_a_ms'].notnull() & df['delay_b_ms'].notnull()]

    df['dur_ms'] = df['delay_b_ms'] + df['delay_a_ms']
    df['dur_ms_rounded'] = df['dur_ms'].apply(lambda x: np.round(x, decimals=0))

    df['linear_ratio'] = df['delay_b_ms'] / (df['delay_a_ms'] + df['delay_b_ms'])
    df['ratio'] = df['linear_ratio']  # np.log10(df['linear_ratio'])
    df['ratio_rounded'] = df['ratio'].apply(lambda x: np.round(x, decimals=2))
    df = df[df['round'] >= min_round]  # 20 rounds are approximatly 10 minutes after start of the experiment
    return df

# alpha=0.01 corresponds to 99% confidence interval
def calc_ci_of_sd(sd, num, alpha=0.01):
    low = np.sqrt(((num-1)*(sd**2))/scipy.stats.chi2.ppf(1.0-alpha/2.0, num - 1))
    up = np.sqrt(((num-1)*(sd**2))/scipy.stats.chi2.ppf(alpha/2.0, num - 1))
    return (low, up)

max_slot_durs = [42]

logfiles = [
        '2024-02-28_ping_pong_200/job_11985.tar.gz',
        '2024-02-28_ping_pong_200/job_11986.tar.gz',
        '2024-02-28_ping_pong_200/job_11987.tar.gz',
        '2024-02-28_ping_pong_200/job_11988.tar.gz',
        '2024-02-28_ping_pong_200/job_11989.tar.gz',
        '2024-02-28_ping_pong_200/job_11990.tar.gz',
        '2024-02-28_ping_pong_200/job_11991.tar.gz',
        '2024-02-28_ping_pong_200/job_11992.tar.gz',
        '2024-02-28_ping_pong_200/job_11993.tar.gz',
        '2024-02-28_ping_pong_200/job_11994.tar.gz',
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

def export_delay_exp_ping_pong(export_dir):
    assume_twr_equal_noise = False

    min_round = 50

    initiator = 3
    responders = [0, 1, 2, 4, 5, 6]
    passive_devs = [0, 1, 2, 4, 5]

    overall_r2_fits = {}
    for max_slots_dur in [42]: #list(range(18, 42+1, 8)):
        print("################# MAX SLOTS DUR", max_slots_dur)
        all_res = 0.0
        all_tot = 0.0

        import export_drift_rate
        rx_noise_map = export_drift_rate.estimate_reception_noise_map(None, use_bias_correction=True, min_round=min_round)
        for use_rx_noise_est in [True]:
            with open('{}/ping_pong_std_initiator_fit_passive_{}_{}_slot_dur_{}.csv'.format(export_dir, use_rx_noise_est, initiator, max_slots_dur), 'w', newline='') as csvfile:

                writer = csv.writer(csvfile)


                row = [
                    'responder', 'mae', 'fitted_noise_initiator', 'fitted_noise_responder', 'r2', 'r2_alt'
                ]

                for d in passive_devs:
                    row += [
                        'mae_passive_{}'.format(d), 'fitted_noise_initiator_passive_{}'.format(d), 'fitted_noise_responder_passive_{}'.format(d), 'r2_passive_{}'.format(d)
                    ]
                writer.writerow(row)

                for responder in responders:
                    row = [responder]

                    # if False:
                    #     for max_slots_dur in range(2, 52, 2):
                    #
                    #
                    #         dfs = in_parallel(
                    #             [functools.partial(get_df, log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles]
                    #         )
                    #
                    #
                    #         continue
                    #         active_df = pd.concat(dfs, ignore_index=True, copy=True)
                    #
                    #         active_df = prepare_df(active_df)
                    #
                    #         fig, ax = plt.subplots()
                    #
                    #         aggr = active_df.groupby('init_slot').agg(
                    #             {
                    #                 'twr_tof_ds_err': 'std',
                    #                 'twr_tof_ss_err': 'std',
                    #                 'twr_tof_ss_reverse_err': 'std',
                    #                 'twr_tof_ss_avg': 'std',
                    #                 'linear_ratio': 'mean',
                    #                 'ratio_rounded': 'count',
                    #                 'init_slot': 'mean'
                    #             }
                    #         )
                    #
                    #         aggr.plot.line('init_slot', 'twr_tof_ds_err', alpha=0.5, ax=ax, label='twr_tof_ds_err std', c='C0')
                    #         plt.show()
                    #         exit()
                    #
                    #
                    # if False:
                    #
                    #     dfs = [
                    #         get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles for max_slots_dur in range(6, 201, 4)
                    #     ]
                    #
                    #     print(dfs)
                    #
                    #     active_df = pd.concat(dfs, ignore_index=True, copy=True)
                    #
                    #     active_df = prepare_df(active_df)
                    #
                    #     fig, ax = plt.subplots()
                    #
                    #     def aggr(df):
                    #         return df.groupby('dur_ms_rounded').agg(
                    #             {
                    #                 'twr_tof_ds_err': 'std',
                    #                 'twr_tof_ss_err': 'std',
                    #                 'twr_tof_ss_reverse_err': 'std',
                    #                 'twr_tof_ss_avg': 'std',
                    #                 'linear_ratio': 'mean',
                    #                 'ratio_rounded': 'count',
                    #                 'dur_ms_rounded': 'mean'
                    #             }
                    #         )
                    #
                    #     df_15 = active_df[active_df['ratio_rounded'] <= 0.15]
                    #     df_center = active_df[(active_df['ratio_rounded'] > 0.15) & (active_df['ratio_rounded'] < 0.85)]
                    #     df_85 = active_df[active_df['ratio_rounded'] >= 0.85]
                    #     df_mean = active_df[active_df['ratio_rounded'] == 0.50]
                    #
                    #     df_15 = aggr(df_15)
                    #     df_center = aggr(df_center)
                    #     df_85 = aggr(df_85)
                    #     df_mean = aggr(df_mean)
                    #
                    #     df_15.plot.line('dur_ms_rounded', 'twr_tof_ds_err', alpha=0.5, ax=ax, label='<= 15%', c='C0')
                    #     df_center.plot.line('dur_ms_rounded', 'twr_tof_ds_err', alpha=0.5, ax=ax, label='>15% < 85%', c='C1')
                    #     df_85.plot.line('dur_ms_rounded', 'twr_tof_ds_err', alpha=0.5, ax=ax, label='>= 85%', c='C2')
                    #     df_mean.plot.line('dur_ms_rounded', 'twr_tof_ds_err', alpha=0.5, ax=ax, label='mean', c='C3')
                    #
                    #     plt.show()
                    #
                    #     exit()


                    dfs = [
                        get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles
                    ]

                    active_df = pd.concat(dfs, ignore_index=True, copy=True)

                    active_df = prepare_df(active_df, initiator=initiator, responder=responder, min_round=min_round)


                    # 3,4, 3to4
                    # 5, 4 to 5
                    # 6 3.5 to 4.5
                    # 7 6to8
                    # 8, 3.5 to 4.5
                    # 9 3.5 to 4.5
                    # 10 3.5 to 4.5
                    # 12 3 to 4
                    if len(passive_devs):
                        dfs = [
                            get_df(log, tdoa_src_dev_number=d, max_slots_dur=max_slots_dur) for log in logfiles for d in passive_devs
                        ]
                        passive_df = pd.concat(dfs, ignore_index=True, copy=True)
                        passive_df = prepare_df(passive_df, initiator=initiator, responder=responder, min_round=min_round)
                    else:
                        passive_df = None


                    # TODO add passive_df!!
                    # active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
                    #                                                           use_bias_correction=True, skip_to_round=0,
                    #                                                           up_to_round=None)

                    me = active_df['twr_tof_ds_err'].mean()
                    mae = np.mean(np.abs(active_df['twr_tof_ds_err']))
                    row += [mae]

                    # active_df = active_df[active_df['pair'] == "0-3"]
                    # print(active_df['ratio_rounded'].unique())
                    active_df_aggr = active_df.groupby('ratio_rounded').agg(
                        {
                            'twr_tof_ds_err': 'std',
                            'twr_tof_ss_err': 'std',
                            'twr_tof_ss_reverse_err': 'std',
                            'twr_tof_ss_avg': 'std',
                            'linear_ratio': 'mean',
                            'ratio_rounded': 'count'
                        }
                    )

                    # we filter out ratios with less than 2 samples
                    active_df_aggr = active_df_aggr[active_df_aggr['ratio_rounded'] > 1]


                    # active_df_aggr.plot.scatter(x=active_df_aggr['ratio_rounded'], y='twr_tof_ds_err')
                    # TODO: Fit curve?!
                    resp_delay_s = 0.002

                    def calc_delays_from_linear(linear_ratio):
                        if linear_ratio >= 1:
                            delay_a = resp_delay_s
                            delay_b = resp_delay_s * linear_ratio
                        else:
                            delay_a = resp_delay_s * (1.0 / linear_ratio)
                            delay_b = resp_delay_s
                        return delay_b, delay_a

                    def calc_delays_from_exp(exp_ratio):
                        if exp_ratio >= 0:
                            delay_a = resp_delay_s
                            delay_b = resp_delay_s * np.power(10, exp_ratio)
                        else:
                            delay_a = resp_delay_s * np.power(10, -exp_ratio)
                            delay_b = resp_delay_s
                        return delay_b, delay_a

                    # we fit a curve to the TWR measurements

                    num_per_bin = 0
                    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C5', 'C6']

                    if assume_twr_equal_noise:
                        def calc_predicted_tof_std(linear_ratios, a_b_std):
                            b_a_std = a_b_std
                            return np.sqrt(
                                (0.5 * b_a_std) ** 2
                                + (0.5 * linear_ratios * a_b_std) ** 2
                                + (0.5 * (1 - linear_ratios) * a_b_std) ** 2
                            )
                    else:
                        def calc_predicted_tof_std(linear_ratios, a_b_std, b_a_std):
                            return np.sqrt(
                                (0.5 * b_a_std) ** 2
                                + (0.5 * linear_ratios * a_b_std) ** 2
                                + (0.5 * (1-linear_ratios) * a_b_std) ** 2
                            )

                    data_xs = active_df_aggr['linear_ratio'].to_numpy()
                    data_xs_ratio = data_xs# np.round(np.log10(data_xs) * 100) / 100.0
                    data_ys = active_df_aggr['twr_tof_ds_err'].to_numpy()
                    data_counts = active_df_aggr['ratio_rounded'].to_numpy()

                    popt, pcov = scipy.optimize.curve_fit(calc_predicted_tof_std, data_xs, data_ys, bounds=(0, 2))

                    alt_pred = None
                    alt_r2_fit = None
                    if use_rx_noise_est:
                        popt = [rx_noise_map[(initiator, responder)], rx_noise_map[(responder, initiator)]]

                        delay_b = data_xs*max_slots_dur*0.0075
                        delay_a = (1.0-data_xs)*max_slots_dur*0.0075

                        n = np.mean(popt)
                        alt_pred = calc_predicted_tof_std_navratil(n,n, delay_b, delay_a)
                        alt_residuals = data_ys - alt_pred
                        print(alt_pred)

                        alt_ss_res = np.sum(alt_residuals ** 2)
                        alt_ss_tot = np.sum((data_ys - np.mean(data_ys)) ** 2)

                        alt_r2_fit = np.round(1 - (alt_ss_res / alt_ss_tot), 3)

                        print("Alt Pred R2 Fit", alt_r2_fit)

                    if len(popt) > 1:
                        pred_twr_ys = calc_predicted_tof_std(data_xs, popt[0], popt[1])
                        residuals = data_ys - calc_predicted_tof_std(data_xs, *popt)
                    else:
                        pred_twr_ys = calc_predicted_tof_std(data_xs, popt[0])
                        residuals = data_ys - calc_predicted_tof_std(data_xs, *popt)
                        popt = [popt[0], popt[0]] # we pretend that we are still fitting two parameters



                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((data_ys - np.mean(data_ys)) ** 2)

                    all_res += ss_res
                    all_tot += ss_tot
                    print("Active R2 Fit", np.round(1 - (ss_res / ss_tot), 3), 1 - (ss_res / ss_tot))
                    row += [popt[0], popt[1]]
                    row += [np.round(1 - (ss_res / ss_tot), 3)]
                    row += [alt_r2_fit]

                    print("Optimal TWR Fit", popt)
                    print("NUM DATAPOINTS overall", len(active_df))
                    print("NUM ratios", len(active_df_aggr))



                    def calc_predicted_tdoa_std(linear_ratios, a_b_std, b_a_std, a_p_std, b_p_std):

                        return np.sqrt(
                            (0.5 * b_a_std) ** 2
                            + (0.5 * a_b_std * (linear_ratios - 1)) ** 2
                            + (0.5 * a_b_std * (linear_ratios)) ** 2
                            + (a_p_std * (1 - linear_ratios)) ** 2
                            + b_p_std ** 2
                            + (a_p_std * (linear_ratios)) ** 2
                        )

                    fig, ax = plt.subplots()
                    plt.plot(data_xs_ratio, pred_twr_ys, alpha=0.5, linestyle='--', color=colors[0])

                    if alt_pred is not None:
                        plt.plot(data_xs_ratio, alt_pred, alpha=0.5, linestyle='--', color='gray', label='DS-TWR Pred\n[Navrátil and Vejražka]')

                    # label='Fit ToF $(\sigma_{{AB}}={:.2f}, \sigma_{{BA}}={:.2f})$'.format(popt[0]*100, popt[1]*100))

                    def scatter_bins(df, col, color):
                        if num_per_bin == 0:
                            return
                        xs = []
                        ys = []

                        for name, group in df.groupby('ratio_rounded'):
                            r = group['ratio_rounded'].to_numpy()[0]
                            l = group[col].to_numpy()
                            ls = np.array_split(l, indices_or_sections=len(l) / num_per_bin)

                            ys += [np.std(x) for x in ls]
                            xs += [r for x in ls]

                        plt.scatter(x=xs, y=ys, c=color, s=2.5)

                    scatter_bins(active_df, 'twr_tof_ds_err', colors[0])
                    active_df_aggr.plot.line(y='twr_tof_ds_err', ax=ax, label="ToF SD", style='-', color=colors[0])

                    ci_cd_low, ci_cd_high = calc_ci_of_sd(data_ys, data_counts)

                    #print(ci_cd_low, ci_cd_high)

                    plt.fill_between(data_xs, ci_cd_low, ci_cd_high, color=colors[0], alpha=0.25)


                    #active_df_aggr.plot.line(y='twr_tof_ss_err', ax=ax)
                    #active_df_aggr.plot.line(y='twr_tof_ss_reverse_err', ax=ax)
                    #active_df_aggr.plot.line(y='twr_tof_ss_avg', ax=ax)

                    for (i, passive_dev) in enumerate(passive_devs):
                        if passive_dev in [initiator, responder]:
                            row += [''] * 4 # we add empty entries in case the passive device was involved in the active measurement
                            continue

                        filt_df = passive_df[passive_df['tdoa_device'] == passive_dev]

                        me = filt_df['tdoa_est_ds_err'].mean()
                        mae = np.mean(np.abs(filt_df['tdoa_est_ds_err']))
                        row += [mae]

                        aggr_filt_df = filt_df.groupby('ratio_rounded').agg(
                            {
                                'tdoa_est_ds': 'std',
                                'linear_ratio': 'mean',
                                'ratio_rounded': 'count',
                                'tdoa_est_mixed': 'std',
                                'tdoa_est_ss_init': 'std'
                            }
                        )

                        aggr_filt_df = aggr_filt_df[aggr_filt_df['ratio_rounded'] > 1]

                        data_xs = aggr_filt_df['linear_ratio'].to_numpy()
                        data_xs_ratio = data_xs# np.round(np.log10(data_xs) * 100) / 100.0
                        data_ys = aggr_filt_df['tdoa_est_ds'].to_numpy()

                        data_counts = aggr_filt_df['ratio_rounded'].to_numpy()

                        def fit(linear_ratios, a_p_std, b_p_std):
                            return calc_predicted_tdoa_std(linear_ratios, popt[0], popt[1], a_p_std, b_p_std)

                        if use_rx_noise_est == False:
                            try:
                                passive_popt, passive_pcov = scipy.optimize.curve_fit(fit, data_xs, data_ys, bounds=(0, 2))
                            except Exception as e:
                                print("Error", e)
                                passive_popt, passive_pcov = ([0.0, 0.0], [])
                        else:
                            #we should have the values from the first iteration
                            passive_popt = [rx_noise_map[(initiator, passive_dev)], rx_noise_map[(responder, passive_dev)]]

                        residuals = data_ys - fit(data_xs, *passive_popt)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((data_ys - np.mean(data_ys)) ** 2)

                        all_res += ss_res
                        all_tot += ss_tot

                        print("Passive {}".format(i), np.round(1 - (ss_res / ss_tot), 3), 1 - (ss_res / ss_tot))

                        pred_tdoa_ys = calc_predicted_tdoa_std(data_xs, popt[0], popt[1], passive_popt[0],
                                                               passive_popt[1])
                        row += [passive_popt[0], passive_popt[1]]


                        row += [np.round(1 - (ss_res / ss_tot), 3)]

                        print("Optimal TDoA Fit", i, popt[0], popt[1], passive_popt[0], passive_popt[1], np.round(popt[0] * 100, 2),
                              np.round(popt[1] * 100, 2), np.round(passive_popt[0] * 100, 2), np.round(passive_popt[1] * 100, 2))

                        scatter_bins(filt_df, 'tdoa_est_ds', colors[i + 1])

                        ci_cd_low, ci_cd_high = calc_ci_of_sd(data_ys, data_counts)
                        plt.fill_between(data_xs_ratio, ci_cd_low, ci_cd_high, color=colors[i + 1], alpha=0.25)

                        plt.plot(data_xs_ratio, pred_tdoa_ys, color=colors[i + 1], linestyle='--', alpha=0.5)
                        # label='Fit TDoA $(\sigma_{{AL}}={:.2f}, \sigma_{{BL}}={:.2f})$'.format(passive_popt[0]*100, passive_popt[1]*100),



                        aggr_filt_df.plot.line(y='tdoa_est_ds', ax=ax, label="TDoA $L{}$ SD".format(i + 1), color=colors[i + 1],
                                               style='-')

                    def formatter(x):
                        delay_b, delay_a = calc_delays_from_exp(x)
                        delay_a = round(delay_a * 1000)
                        delay_b = round(delay_b * 1000)
                        return r'${{{}}}:{{{}}}$'.format(delay_b, delay_a)

                    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
                    ax.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm
                    fig.set_size_inches(6.0, 6.0)

                    ax.set_xlabel('Delay Ratio')
                    ax.set_ylabel('SD [cm]')

                    #ax.set_ylim([0.0, 0.05])

                    plt.grid(color='lightgray', linestyle='dashed')

                    plt.legend(reverse=True)
                    # plt.tight_layout()

                    # ax.set_ylim([0.0, 0.25])
                    # ax.set_xlim([-100.0, +100.0])

                    plt.savefig("{}/std_fit_ping_pong_{}_750usec_initiator_{}_responder_{}_use_rx_noise_{}.pdf".format(export_dir, max_slots_dur, initiator, responder, use_rx_noise_est), bbox_inches='tight')  # , pad_inches=0)

                    plt.close()
                    writer.writerow(row)
        overall_r2_fits[max_slots_dur] = np.round(1 - (ss_res / ss_tot), 3)
        print("OVERALL R2 Fit", max_slots_dur, overall_r2_fits[max_slots_dur])
    print("OVERALL R2 FITS")
    print(overall_r2_fits)

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_delay_exp_ping_pong(config['EXPORT_DIR'])




