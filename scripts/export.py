import os
import progressbar
import numpy as np
import json

import scipy.optimize

import logs
import utility

from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, gen_delay_estimates_from_testbed_run
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd


import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached_legacy, init_cache, load_env_config, set_global_cache_prefix_by_config
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import pandas as pd

METHOD_PREFIX = 'export_'

CONFIDENCE_FILL_COLOR = '0.8'
PERCENTILES_FILL_COLOR = '0.5'
COLOR_MAP = 'tab10'

c_in_air = 299702547.236


PROTOCOL_NAME = "X"


def load_plot_defaults():
    # Configure as needed
    plt.rc('lines', linewidth=2.0)
    #plt.rc('image', cmap='viridis')
    plt.rc('legend', framealpha=1.0, fancybox=True)
    plt.rc('errorbar', capsize=3)
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', size=11)
    #plt.rc('font', size=8, family="serif", serif=['Times New Roman'] + plt.rcParams['font.serif'])
    plt.rcParams['axes.axisbelow'] = True


import subprocess
import os


DEFAULT_CROP = False

def save_and_crop(path, *args, **kwargs):

    filename, file_extension = os.path.splitext(path)

    crop = kwargs.pop('crop', DEFAULT_CROP)
    plt.savefig(path, *args, **kwargs)

    if crop:
        if file_extension == ".pdf":
            cropped_path = filename + "_cropped" + file_extension
            subprocess.run(["pdfcrop", path, cropped_path], stdout=subprocess.DEVNULL)

def add_df_cols(df, tdoa_src_dev_number=None):

    df['twr_tof_ds_err'] = df['twr_tof_ds'] - df['dist']
    df['twr_tof_ss_err'] = df['twr_tof_ss'] - df['dist']
    df['twr_tof_ss_reverse_err'] = df['twr_tof_ss_reverse'] - df['dist']

    if tdoa_src_dev_number is not None:
        df['tdoa_est_ds_err'] = df['tdoa_est_ds'] - df['tdoa']
        df['tdoa_est_ss_init_err'] = df['tdoa_est_ss_init'] - df['tdoa']
        df['tdoa_est_ss_final_err'] = df['tdoa_est_ss_final'] - df['tdoa']
        df['tdoa_est_ss_both_err'] = df['tdoa_est_ss_both'] - df['tdoa']
        df['tdoa_est_mixed_err'] = df['tdoa_est_mixed'] - df['tdoa']

    return df


if __name__ == '__main__':

    config = load_env_config()

    load_plot_defaults()

    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']

    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])
