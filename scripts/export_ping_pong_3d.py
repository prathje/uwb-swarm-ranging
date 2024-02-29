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
        # '2024-02-28_ping_pong_200/job_11986.tar.gz',
        # '2024-02-28_ping_pong_200/job_11987.tar.gz',
        # '2024-02-28_ping_pong_200/job_11988.tar.gz',
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

def export_ping_pong_3d( export_dir):

        dfs = [
            get_df(log, tdoa_src_dev_number=None, max_slots_dur=max_slots_dur) for log in logfiles for max_slots_dur in max_slot_durs
        ]

        active_df = pd.concat(dfs, ignore_index=True, copy=True)
        active_df = prepare_df(active_df)



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

        active_df_aggr = active_df.groupby(['dur_ms_rounded', 'ratio_rounded']).agg(
            {
                'twr_tof_ds_err': 'std',
                'twr_tof_ss_err': 'std',
                'twr_tof_ss_reverse_err': 'std',
                'twr_tof_ss_avg': 'std',
                'linear_ratio': 'mean',
                'delay_b_ms_rounded': 'mean',
                'delay_a_ms_rounded': 'mean',
                'dur_ms_rounded': 'mean',
                'ratio_rounded': 'mean',
            }
        )

        # we filter out ratios with less than 2 samples
        #active_df_aggr = active_df_aggr[active_df_aggr['delay_b_ms_rounded'] > 1]
        print(active_df_aggr)

        ax = plt.axes(projection='3d')

        x = active_df_aggr['dur_ms_rounded'].to_numpy()
        y = active_df_aggr['ratio_rounded'].to_numpy()
        z = active_df_aggr['twr_tof_ds_err'].to_numpy()

        ax.scatter(x, y, z, alpha=0.6, c=z, cmap='viridis')
        #ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
        plt.show()

if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    export_ping_pong_3d(config['EXPORT_DIR'])




