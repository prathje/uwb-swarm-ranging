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
        # '2024-03-01_ping_pong_200/job_12012.tar.gz'
]

initiator = 3
responder = 0
passive_dev = 1
max_slot_dur = 10

def export_ds_cfo_active_std_comparison(export_dir):
    dfs = [
        get_df(log, tdoa_src_dev_number=passive_dev, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    active_df = pd.concat(dfs, ignore_index=True, copy=True)

    active_df = prepare_df(active_df, initiator=initiator, responder=responder)

    active_df['twr_tof_ss_avg'] = ((active_df['twr_tof_ss'] + active_df['twr_tof_ss_reverse']) / 2)
    active_df['twr_tof_ss_avg_err'] = active_df['twr_tof_ss_avg'] - active_df['dist']

    # active_df = active_df[active_df['pair'] == "0-3"]
    # print(active_df['ratio_rounded'].unique())
    active_df_aggr = active_df.groupby('ratio_rounded').agg(
        {
            'twr_tof_ds_err': 'std',
            'twr_tof_ss_err': 'std',
            'twr_tof_ss_reverse_err': 'std',
            'twr_tof_ss_avg_err': 'std',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C5', 'C6']
    resp_delay_s = 0.002
    def calc_delays_from_exp(exp_ratio):
        if exp_ratio >= 0:
            delay_a = resp_delay_s
            delay_b = resp_delay_s * np.power(10, exp_ratio)
        else:
            delay_a = resp_delay_s * np.power(10, -exp_ratio)
            delay_b = resp_delay_s
        return delay_b, delay_a
    def formatter(x):
        print(x)
        delay_b, delay_a = calc_delays_from_exp(x)
        delay_a = round(delay_a * 1000)
        delay_b = round(delay_b * 1000)
        return r'${{{}}}:{{{}}}$'.format(delay_b, delay_a)


    fig, ax = plt.subplots()
    active_df_aggr.plot.line(y='twr_tof_ds_err', ax=ax, label="DS-TWR", style='-', color=colors[0])
    active_df_aggr.plot.line(y='twr_tof_ss_avg_err', ax=ax, label="SS-TWR (AVG)", style='-', color=colors[1])
    active_df_aggr.plot.line(y='twr_tof_ss_reverse_err', ax=ax, label="SS-TWR $(B)$", style='-', color=colors[2])
    active_df_aggr.plot.line(y='twr_tof_ss_err', ax=ax, label="SS-TWR $(A)$ ", style='-', color=colors[3])


    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm
    fig.set_size_inches(3.0, 2.5)

    ax.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax.set_ylabel('Mean SD [cm]')

    plt.grid(color='lightgray', linestyle='dashed')

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    plt.legend(reverse=True)
    # plt.tight_layout()

    #ax.set_xlim([-0.6, +0.6])

    #ax.set_ylim([0.016, 0.05])
    save_and_crop("{}/std_active.pdf".format(export_dir), bbox_inches='tight')  # , pad_inches=0)

    plt.close()

def export_ds_cfo_passive_std_comparison(export_dir):

    dfs = [
        get_df(log, tdoa_src_dev_number=passive_dev, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    passive_df = pd.concat(dfs, ignore_index=True, copy=True)

    passive_df = prepare_df(passive_df, initiator=initiator, responder=responder)

    # passive_df = passive_df[passive_df['pair'] == "0-3"]
    # print(passive_df['ratio_rounded'].unique())
    passive_df_aggr = passive_df.groupby('ratio_rounded').agg(
        {
            'tdoa_est_ds': 'std',
            'tdoa_est_mixed': 'std',
            'tdoa_est_ss_init': 'std',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C5', 'C6']
    resp_delay_s = 0.002
    def calc_delays_from_exp(exp_ratio):
        if exp_ratio >= 0:
            delay_a = resp_delay_s
            delay_b = resp_delay_s * np.power(10, exp_ratio)
        else:
            delay_a = resp_delay_s * np.power(10, -exp_ratio)
            delay_b = resp_delay_s
        return delay_b, delay_a
    def formatter(x):
        print(x)
        delay_b, delay_a = calc_delays_from_exp(x)
        delay_a = round(delay_a * 1000)
        delay_b = round(delay_b * 1000)
        return r'${{{}}}:{{{}}}$'.format(delay_b, delay_a)


    fig, ax = plt.subplots()
    passive_df_aggr.plot.line(y='tdoa_est_ds', ax=ax, label="DS-TDoA", style='-', color=colors[0])
    passive_df_aggr.plot.line(y='tdoa_est_mixed', ax=ax, label="Mixed TDoA", style='-', color=colors[1])
    passive_df_aggr.plot.line(y='tdoa_est_ss_init', ax=ax, label="SS-TDoA", style='-', color=colors[2])


    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm
    fig.set_size_inches(3.0, 2.5)

    ax.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax.set_ylabel('Mean SD [cm]')


    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(reverse=True)
    # plt.tight_layout()

    #ax.set_xlim([-0.6, +0.6])


    #ax.set_ylim([0.03, 0.05])
    save_and_crop("{}/std_passive.pdf".format(export_dir), bbox_inches='tight')  # , pad_inches=0)

    plt.close()


def export_ds_cfo_active_mae_comparison(export_dir):
    dfs = [
        get_df(log, tdoa_src_dev_number=passive_dev, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    active_df = pd.concat(dfs, ignore_index=True, copy=True)

    active_df = prepare_df(active_df)

    active_df['twr_tof_ss_avg'] = ((active_df['twr_tof_ss'] + active_df['twr_tof_ss_reverse']) / 2)
    active_df['twr_tof_ss_avg_err'] = active_df['twr_tof_ss_avg'] - active_df['dist']

    active_df['twr_tof_ds_err'] = active_df['twr_tof_ds_err'].apply(np.abs)
    active_df['twr_tof_ss_err'] = active_df['twr_tof_ss_err'].apply(np.abs)
    active_df['twr_tof_ss_reverse_err'] = active_df['twr_tof_ss_reverse_err'].apply(np.abs)
    active_df['twr_tof_ss_avg_err'] = active_df['twr_tof_ss_avg_err'].apply(np.abs)

    # active_df = active_df[active_df['pair'] == "0-3"]
    # print(active_df['ratio_rounded'].unique())
    active_df_aggr = active_df.groupby('ratio_rounded').agg(
        {
            'twr_tof_ds_err': 'mean',
            'twr_tof_ss_err': 'mean',
            'twr_tof_ss_reverse_err': 'mean',
            'twr_tof_ss_avg_err': 'mean',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C5', 'C6']
    resp_delay_s = 0.002
    def calc_delays_from_exp(exp_ratio):
        if exp_ratio >= 0:
            delay_a = resp_delay_s
            delay_b = resp_delay_s * np.power(10, exp_ratio)
        else:
            delay_a = resp_delay_s * np.power(10, -exp_ratio)
            delay_b = resp_delay_s
        return delay_b, delay_a
    def formatter(x):
        print(x)
        delay_b, delay_a = calc_delays_from_exp(x)
        delay_a = round(delay_a * 1000)
        delay_b = round(delay_b * 1000)
        return r'${{{}}}:{{{}}}$'.format(delay_b, delay_a)


    fig, ax = plt.subplots()
    active_df_aggr.plot.line(y='twr_tof_ds_err', ax=ax, label="DS-TWR", style='-', color=colors[0])
    active_df_aggr.plot.line(y='twr_tof_ss_avg_err', ax=ax, label="SS-TWR (AVG)", style='-', color=colors[1])
    active_df_aggr.plot.line(y='twr_tof_ss_reverse_err', ax=ax, label="SS-TWR $(B)$", style='-', color=colors[2])
    active_df_aggr.plot.line(y='twr_tof_ss_err', ax=ax, label="SS-TWR $(A)$ ", style='-', color=colors[3])


    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm
    fig.set_size_inches(6.0, 4.5)

    ax.set_xlabel('Delay Ratio [ms : ms]')
    ax.set_ylabel('Mean Absolute Error [cm]')

    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(reverse=True)
    # plt.tight_layout()

    #ax.set_xlim([-0.6, +0.6])

    #ax.set_ylim([0.016, 0.036])
    save_and_crop("{}/mean_active.pdf".format(export_dir), bbox_inches='tight')  # , pad_inches=0)

    plt.close()

def export_ds_cfo_passive_mae_comparison(export_dir):

    dfs = [
        get_df(log, tdoa_src_dev_number=passive_dev, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    passive_df = pd.concat(dfs, ignore_index=True, copy=True)

    passive_df = prepare_df(passive_df)

    passive_df['tdoa_est_ds_err'] = passive_df['tdoa_est_ds_err'].apply(np.abs)
    passive_df['tdoa_est_mixed_err'] = passive_df['tdoa_est_mixed_err'].apply(np.abs)
    passive_df['tdoa_est_ss_init_err'] = passive_df['tdoa_est_ss_init_err'].apply(np.abs)

    # passive_df = passive_df[passive_df['pair'] == "0-3"]
    # print(passive_df['ratio_rounded'].unique())
    passive_df_aggr = passive_df.groupby('ratio_rounded').agg(
        {
            'tdoa_est_ds_err': 'mean',
            'tdoa_est_mixed_err': 'mean',
            'tdoa_est_ss_init_err': 'mean',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C5', 'C6']
    resp_delay_s = 0.002
    def calc_delays_from_exp(exp_ratio):
        if exp_ratio >= 0:
            delay_a = resp_delay_s
            delay_b = resp_delay_s * np.power(10, exp_ratio)
        else:
            delay_a = resp_delay_s * np.power(10, -exp_ratio)
            delay_b = resp_delay_s
        return delay_b, delay_a
    def formatter(x):
        print(x)
        delay_b, delay_a = calc_delays_from_exp(x)
        delay_a = round(delay_a * 1000)
        delay_b = round(delay_b * 1000)
        return r'${{{}}}:{{{}}}$'.format(delay_b, delay_a)


    fig, ax = plt.subplots()
    passive_df_aggr.plot.line(y='tdoa_est_ds_err', ax=ax, label="DS-TDoA", style='-', color=colors[0])
    passive_df_aggr.plot.line(y='tdoa_est_mixed_err', ax=ax, label="Mixed TDoA", style='-', color=colors[1])
    passive_df_aggr.plot.line(y='tdoa_est_ss_init_err', ax=ax, label="SS-TDoA", style='-', color=colors[2])


    #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm
    fig.set_size_inches(6.0, 4.5)

    ax.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax.set_ylabel('MAE [cm]')

    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(reverse=True)
    # plt.tight_layout()

    #ax.set_xlim([-0.6, +0.6])

    #ax.set_ylim([0.03, 0.08])
    save_and_crop("{}/mae_passive_close.pdf".format(export_dir), bbox_inches='tight')  # , pad_inches=0)

    plt.close()


def export_passive_combined(export_dir):


    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 1], sharey=True)
    fig.subplots_adjust(hspace=0.01)


    dfs = [
        get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    active_df = pd.concat(dfs, ignore_index=True, copy=True)

    active_df = prepare_df(active_df, initiator=initiator, responder=responder)

    active_df['twr_tof_ss_avg'] = ((active_df['twr_tof_ss'] + active_df['twr_tof_ss_reverse']) / 2)
    active_df['twr_tof_ss_avg_err'] = active_df['twr_tof_ss_avg'] - active_df['dist']

    # active_df = active_df[active_df['pair'] == "0-3"]
    # print(active_df['ratio_rounded'].unique())
    active_df_aggr = active_df.groupby('ratio_rounded').agg(
        {
            'twr_tof_ds_err': 'std',
            'twr_tof_ss_err': 'std',
            'twr_tof_ss_reverse_err': 'std',
            'twr_tof_ss_avg_err': 'std',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C6', 'C7']
    markers = ["o", ".", "v", "^", "+", "*", "s"]
    marker_size = 5


    active_df_aggr.plot.line(y='twr_tof_ds_err', ax=ax1, label="DS-TWR", style='-', color=colors[0], marker=markers[0],ms=marker_size)
    active_df_aggr.plot.line(y='twr_tof_ss_avg_err', ax=ax1, label="SS-TWR (AVG)", style='-', color=colors[1], marker=markers[1],ms=marker_size)
    active_df_aggr.plot.line(y='twr_tof_ss_reverse_err', ax=ax1, label="SS-TWR $(B)$", style='-', color=colors[2], marker=markers[2],ms=marker_size)
    active_df_aggr.plot.line(y='twr_tof_ss_err', ax=ax1, label="SS-TWR $(A)$ ", style='-', color=colors[3], marker=markers[3],ms=marker_size)

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax1.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm

    ax1.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax1.set_ylabel('Mean SD [cm]')

    ax1.grid(color='lightgray', linestyle='dashed')

    ax1.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    ax1.legend(reverse=True)
    # plt.tight_layout()

    # ax.set_xlim([-0.6, +0.6])

    dfs = [
        get_df(log, tdoa_src_dev_number=passive_dev, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    passive_df = pd.concat(dfs, ignore_index=True, copy=True)

    passive_df = prepare_df(passive_df, initiator=initiator, responder=responder)

    # passive_df = passive_df[passive_df['pair'] == "0-3"]
    # print(passive_df['ratio_rounded'].unique())
    passive_df_aggr = passive_df.groupby('ratio_rounded').agg(
        {
            'tdoa_est_ds': 'std',
            'tdoa_est_mixed': 'std',
            'tdoa_est_ss_init': 'std',
            'linear_ratio': 'mean'
        }
    )

    # we fit a curve to the TWR measurements
    passive_df_aggr.plot.line(y='tdoa_est_ds', ax=ax2, label="DS-TDoA", style='-', color=colors[4],  marker=markers[4],ms=marker_size)
    passive_df_aggr.plot.line(y='tdoa_est_mixed', ax=ax2, label="Mixed TDoA", style='-', color=colors[5],  marker=markers[5],ms=marker_size)
    passive_df_aggr.plot.line(y='tdoa_est_ss_init', ax=ax2, label="SS-TDoA", style='-', color=colors[6],  marker=markers[6],ms=marker_size)

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))

    # ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))
    ax2.yaxis.set_major_formatter(lambda x, pos: np.round(x * 100.0, 1))  # scale to cm

    ax2.set_xlabel(r"Delay Ratio $\dfrac{D_B}{D_B+D_A}$")
    ax2.set_ylabel('SD [cm]')

    ax2.grid(color='lightgray', linestyle='dashed')

    ax2.legend(reverse=True)
    # plt.tight_layout()

    # ax.set_xlim([-0.6, +0.6])

    ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

    fig.set_size_inches(6.0, 4.0)
    fig.tight_layout()
    # ax.set_ylim([0.016, 0.05])
    save_and_crop("{}/cfo_cfo_comparison_combined.pdf".format(export_dir), bbox_inches='tight', crop=True)

    plt.close()



if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_passive_combined(config['EXPORT_DIR'])
    #export_ds_cfo_active_std_comparison(config['EXPORT_DIR'])
    #export_ds_cfo_passive_std_comparison(config['EXPORT_DIR'])
    #export_ds_cfo_active_mae_comparison(config['EXPORT_DIR'])
    #export_ds_cfo_passive_mae_comparison(config['EXPORT_DIR'])




