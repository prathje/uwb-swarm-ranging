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


def export_measured_rx_noise(export_dir):

    skip_to_round = 50
    up_to_round = None
    use_bias_correction = True
    log = 'exp_rx_noise_10041'

    # we also directly search for the triples with the lowest rx variance variation

    lowest_var = 100000.0
    lowest_tripel = []

    import export_drift_rate
    rx_noise_map = export_drift_rate.estimate_reception_noise_map(None, use_bias_correction=True, min_round=50)


    t = trento_a
    ma = np.zeros((len(t.devs), len(t.devs)))

    for a in range(len(t.devs)):
        for b in range(len(t.devs)):
            if b != a:
                ma[a, b] = rx_noise_map[(a, b)] * 100
            else:
                ma[a, b] = 0.0 #np.nan

    plt.clf()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.matshow(ma, cmap='viridis', vmin=0, vmax=5)

    for i in range(ma.shape[0]):
        for j in range(ma.shape[1]):
            if i != j:
                e = ma[i, j]
                if e > 100:
                    e = int(ma[i, j])
                else:
                    e = round(ma[i, j], 1)
                s = str(e)
                ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=False)

    ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
    ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
    fig.set_size_inches(3.5, 3.5)
    plt.tight_layout()

    ax.set_xlabel('RX Device')
    ax.set_ylabel('TX Device')

    save_and_crop("{}/rx_noise_sd_cm_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0, crop=True)

    plt.close()


if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])
    # TODO: Implement the logarithmic one again!
    export_measured_rx_noise(config['EXPORT_DIR'])




