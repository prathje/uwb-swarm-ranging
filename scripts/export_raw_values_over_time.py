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

use_bias_correction = True

logfiles = [
        '2024-02-28_ping_pong_200/job_11985.tar.gz',
        #'2024-02-28_ping_pong_200/job_11986.tar.gz',
        #'2024-02-28_ping_pong_200/job_11987.tar.gz',
        #'2024-02-28_ping_pong_200/job_11988.tar.gz',
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
]

max_slot_durs = list(range(2, 201, 4))

def export_raw_values_over_time(export_dir):

    # TODO add passive_df!!
    # active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
    #                                                           use_bias_correction=True, skip_to_round=0,
    #                                                           up_to_round=None)

    # active_df = active_df[active_df['pair'] == "0-3"]
    # print(active_df['ratio_rounded'].unique())
    # active_df_aggr = active_df.groupby(['delay_b_ms_rounded', 'delay_a_ms_rounded']).agg(
    #     {
    #         'twr_tof_ds_err': 'std',
    #         'twr_tof_ss_err': 'std',
    #         'twr_tof_ss_reverse_err': 'std',
    #         'twr_tof_ss_avg': 'std',
    #         'linear_ratio': 'mean',
    #         'delay_b_ms_rounded': 'mean',
    #         'delay_a_ms_rounded': 'mean',
    #     }
    # )
    for max_slots_dur in [4]:
        raw_dfs = [
            get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles
        ]
        all_df = pd.concat(raw_dfs, ignore_index=True, copy=True)

        plt.clf()
        initiator = 3
        for responder in [4]:
            if initiator == responder:
                continue

            active_df = prepare_df(all_df, initiator=initiator, responder=responder, min_round=50)


            errs = active_df['twr_tof_ds_err']
            ss_errs = active_df['twr_tof_ss_err']


            print(np.mean(errs))
            print(np.mean(ss_errs))

            index = list(range(len(errs)))



            plt.scatter(index, errs, label=f"{initiator}-{responder} DS-TWR", alpha=0.25)
            plt.scatter(index, ss_errs, label=f"{initiator}-{responder} SS-TWR", alpha=0.25)

        plt.xlabel("index")
        plt.ylabel("DS-TWR Error [m]")

        plt.legend()

        plt.savefig(os.path.join(export_dir, f"raw_values_over_time_{max_slots_dur}.pdf"))


def export_std_over_duration_increase(export_dir):
    # TODO add passive_df!!
    # active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
    #                                                           use_bias_correction=True, skip_to_round=0,
    #                                                           up_to_round=None)

    # active_df = active_df[active_df['pair'] == "0-3"]
    # print(active_df['ratio_rounded'].unique())
    # active_df_aggr = active_df.groupby(['delay_b_ms_rounded', 'delay_a_ms_rounded']).agg(
    #     {
    #         'twr_tof_ds_err': 'std',
    #         'twr_tof_ss_err': 'std',
    #         'twr_tof_ss_reverse_err': 'std',
    #         'twr_tof_ss_avg': 'std',
    #         'linear_ratio': 'mean',
    #         'delay_b_ms_rounded': 'mean',
    #         'delay_a_ms_rounded': 'mean',
    #     }
    # )





    plt.clf()
    initiator = 3
    for responder in [4]:
        if initiator == responder:
            continue

        raw_dfs = [
            get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slot_dur) for log in logfiles for max_slot_dur in range(6, 201, 4)
        ]
        all_df = pd.concat(raw_dfs, ignore_index=True, copy=True)

        active_df = prepare_df(all_df, initiator=initiator, responder=responder, min_round=50)

        active_df = active_df[active_df['ratio_rounded'] == 0.5]


        errs = active_df['twr_tof_ds_err']
        ss_errs = active_df['twr_tof_ss_err']
        durs = active_df['dur_ms_rounded']

        print(np.mean(errs))
        print(np.mean(ss_errs))

        plt.scatter(durs, ss_errs, label=f"{initiator}-{responder} SS-TWR", alpha=0.25)
        plt.scatter(durs, errs, label=f"{initiator}-{responder} DS-TWR", alpha=0.25)

        plt.xlabel("index")
        plt.ylabel("DS-TWR Error [m]")

        plt.legend()

        plt.savefig(os.path.join(export_dir, f"raw_vs_duration.pdf"))

        plt.clf()

        grouped_df = active_df.groupby(['dur_ms_rounded']).agg(
            {
                'twr_tof_ds_err': 'std',
                'twr_tof_ss_err': 'std',
                'twr_tof_ss_reverse_err': 'std',
                'twr_tof_ss_avg': 'std',
                'dur_ms_rounded': 'mean',
            }
        )

        errs = grouped_df['twr_tof_ds_err']
        ss_errs = grouped_df['twr_tof_ss_err']
        durs = grouped_df['dur_ms_rounded']

        print(np.mean(errs))
        print(np.mean(ss_errs))

        plt.scatter(durs, ss_errs, label=f"{initiator}-{responder} SS-TWR", alpha=0.25)
        plt.scatter(durs, errs, label=f"{initiator}-{responder} DS-TWR", alpha=0.25)

        plt.xlabel("Duration [ms]")
        plt.ylabel("DS-TWR SD [m]")

        plt.legend()

        plt.savefig(os.path.join(export_dir, f"std_vs_duration.pdf"))


def export_raw_values_and_aggregate(export_dir):

    max_slot_dur = 10
    initiator = 3
    responder = 4
    tdoa_src_dev_number = 5
    filter_ratio = True


    colors = ['C4', 'C1', 'C2', 'C5', 'C3', 'C6', 'C7']

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 1], sharey=True)
    fig.subplots_adjust(hspace=0.01)

    dfs = [
        get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    active_df = pd.concat(dfs, ignore_index=True, copy=True)

    dfs = [
        get_df(log, tdoa_src_dev_number=tdoa_src_dev_number, max_slots_dur=max_slot_dur) for log in logfiles
    ]

    passive_df = pd.concat(dfs, ignore_index=True, copy=True)


    df_3_4 = prepare_df(active_df, initiator=3, responder=4)
    df_3_6 = prepare_df(active_df, initiator=3, responder=6)
    df_3_6_passive_0 = prepare_df(passive_df, initiator=3, responder=4)

    if filter_ratio:
        df_3_4 = df_3_4[df_3_4['ratio_rounded'] == 0.5]
        df_3_6 = df_3_6[df_3_6['ratio_rounded'] == 0.5]
        df_3_6_passive_0 = df_3_6_passive_0[df_3_6_passive_0['ratio_rounded'] == 0.5]



    num_samples = 1000

    df_3_4_errs = list(df_3_4['twr_tof_ds_err'])[0:num_samples]
    df_3_6_errs = list(df_3_6['twr_tof_ds_err'])[0:num_samples]
    df_3_6_passive_0_errs = list(df_3_6_passive_0['tdoa_est_ds_err'])[0:num_samples]


    ax1.scatter(list(range(len(df_3_6_passive_0_errs))), df_3_6_passive_0_errs, label=f"DS-TDoA", alpha=1.0, s=2.0, color=colors[2])
    ax1.scatter(list(range(len(df_3_4_errs))), df_3_4_errs, label=f"DS-TWR", alpha=1.0, s=2.0, color=colors[0])
    ax1.scatter(list(range(len(df_3_6_errs))), df_3_6_errs, label=f"DS-TWR Multipath", alpha=1.0, s=2.0, color=colors[1])

    ax1.set_ylabel('Error [m]')
    ax1.set_xlabel('Measurement')

    meas = [np.mean(df_3_4_errs), np.mean(df_3_6_passive_0_errs), np.mean(df_3_6_errs)]
    stds = [
            np.std(df_3_4_errs),
            np.std(df_3_6_passive_0_errs),
            np.std(df_3_6_errs)
        ]

    rects = ax2.bar(
        [
            f"DS-TWR",
            f"DS-TDoA",
            f"DS-TWR\nMultipath",
        ],
        meas,
        yerr=stds,
        color=[colors[0], colors[2], colors[1]]
    )

    ax2.bar_label(rects, padding=3, fontsize=9, label_type='edge',
                 labels=["{:.2f}\n[{:.2f}]".format(np.round(meas[i], 2), np.round(stds[i], 2)) for i
                         in range(len(meas))])

    lgnd = ax1.legend(framealpha=0.75, reverse=True)
    for handle in lgnd.legend_handles:
        handle.set_sizes([8.0])


    ax1.grid(color='lightgray', linestyle='dashed')
    ax2.grid(color='lightgray', linestyle='dashed')


    fig.set_size_inches(6.0, 4.0)
    fig.tight_layout()
    # ax.set_ylim([0.016, 0.05])
    save_and_crop("{}/raw_comparison.pdf".format(export_dir), bbox_inches='tight', crop=True)

    plt.close()


if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_raw_values_and_aggregate(config['EXPORT_DIR'])
    #export_std_over_duration_increase(config['EXPORT_DIR'])
    #export_raw_values_over_time(config['EXPORT_DIR'])
