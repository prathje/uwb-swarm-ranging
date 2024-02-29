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

use_bias_correction = True

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
        '2024-02-28_ping_pong_200/job_11995.tar.gz',
        '2024-02-28_ping_pong_200/job_11996.tar.gz',
        '2024-02-28_ping_pong_200/job_11997.tar.gz',
        '2024-02-28_ping_pong_200/job_11998.tar.gz',
]


def get_df(log, tdoa_src_dev_number, max_slots_dur):
    def proc():
        it = logs.gen_ping_pong_records(trento_a, log, tdoa_src_dev_number=tdoa_src_dev_number,
                                        bias_corrected=use_bias_correction, max_slot_dur=max_slots_dur)
        df = pd.DataFrame.from_records(it)
        return add_df_cols(df, tdoa_src_dev_number)

    return utility.cached_dt_legacy(  # todo: this was 3
        ('extract_job_tdma_ping_pong_4', log, tdoa_src_dev_number, use_bias_correction, max_slots_dur), proc)




def prepare_df(df):
    initiator = 3
    responder = 0

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
    df = df[df['round'] >= 20]  # 20 rounds are approximatly 10 minutes after start of the experiment
    return df



max_slot_durs = list(range(2, 201, 4))

logfiles = [
    #'ping_pong_trento_a_source_4_750usec_slot_job_11928', # these runs are a bit erroneous
    #'ping_pong_trento_a_source_4_750usec_slot_job_11930',  # these runs are a bit erroneous
    #'ping_pong_trento_a_source_4_750usec_slot_job_11931',  # these runs are a bit erroneous
    '2024-02-28_ping_pong_200/job_11985.tar.gz',
    #'2024-02-28_ping_pong_200/job_11986.tar.gz',
    #'2024-02-28_ping_pong_200/job_11987.tar.gz',
    #'2024-02-28_ping_pong_200/job_11988.tar.gz',
    #'2024-02-28_ping_pong_200/job_11989.tar.gz',
    #'2024-02-28_ping_pong_200/job_11990.tar.gz',
    #'2024-02-28_ping_pong_200/job_11991.tar.gz',
    #'2024-02-28_ping_pong_200/job_11992.tar.gz',
    #'2024-02-28_ping_pong_200/job_11993.tar.gz',
    #'2024-02-28_ping_pong_200/job_11994.tar.gz',
    #'2024-02-28_ping_pong_200/job_11995.tar.gz',
    #'2024-02-28_ping_pong_200/job_11996.tar.gz',
    #'2024-02-28_ping_pong_200/job_11997.tar.gz',
    #'2024-02-28_ping_pong_200/job_11998.tar.gz',
]

def export_delay_exp_ping_pong(export_dir):

    assume_twr_equal_noise = True




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

    for max_slots_dur in [10, 14, 18, 22]: #range(10, 201, 4):
        dfs = [
            get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles
        ]

        active_df = pd.concat(dfs, ignore_index=True, copy=True)

        active_df = prepare_df(active_df)
        passive_devs = [1]

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
            passive_df = prepare_df(passive_df)
        else:
            passive_df = None


        # TODO add passive_df!!
        # active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
        #                                                           use_bias_correction=True, skip_to_round=0,
        #                                                           up_to_round=None)


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

        num_per_bin = 16
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


        popt, pcov = scipy.optimize.curve_fit(calc_predicted_tof_std, data_xs, data_ys)
        if len(popt) > 1:
            pred_twr_ys = calc_predicted_tof_std(data_xs, popt[0], popt[1])
            residuals = data_ys - calc_predicted_tof_std(data_xs, *popt)
        else:
            pred_twr_ys = calc_predicted_tof_std(data_xs, popt[0])
            residuals = data_ys - calc_predicted_tof_std(data_xs, *popt)
            popt = [popt[0], popt[0]] # we pretend that we are still fitting two parameters

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((data_ys - np.mean(data_ys)) ** 2)
        print("Active R2", np.round(1 - (ss_res / ss_tot), 3), 1 - (ss_res / ss_tot))

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
        #active_df_aggr.plot.line(y='twr_tof_ss_err', ax=ax)
        #active_df_aggr.plot.line(y='twr_tof_ss_reverse_err', ax=ax)
        #active_df_aggr.plot.line(y='twr_tof_ss_avg', ax=ax)

        for (i, passive_dev) in enumerate(passive_devs):
            filt_df = passive_df[passive_df['tdoa_device'] == passive_dev]

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

            def fit(linear_ratios, a_p_std, b_p_std):
                return calc_predicted_tdoa_std(linear_ratios, popt[0], popt[1], a_p_std, b_p_std)

            try:
                passive_popt, passive_pcov = scipy.optimize.curve_fit(fit, data_xs, data_ys)
            except Exception as e:
                print("Error", e)
                passive_popt, passive_pcov = ([0.0, 0.0], [])
            pred_tdoa_ys = calc_predicted_tdoa_std(data_xs, popt[0], popt[1], passive_popt[0], passive_popt[1])

            residuals = data_ys - fit(data_xs, *passive_popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((data_ys - np.mean(data_ys)) ** 2)
            print("Passive {}".format(i), np.round(1 - (ss_res / ss_tot), 3), 1 - (ss_res / ss_tot))

            print("Optimal TDoA Fit", i, popt[0], popt[1], passive_popt[0], passive_popt[1], np.round(popt[0] * 100, 2),
                  np.round(popt[1] * 100, 2), np.round(passive_popt[0] * 100, 2), np.round(passive_popt[1] * 100, 2))

            scatter_bins(filt_df, 'tdoa_est_ds', colors[i + 1])

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

        plt.savefig("{}/std_fit_ping_pong_{}_750usec.pdf".format(export_dir, max_slots_dur), bbox_inches='tight')  # , pad_inches=0)

        plt.close()

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_delay_exp_ping_pong(config['EXPORT_DIR'])




