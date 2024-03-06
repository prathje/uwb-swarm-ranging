import os
import sys
from multiprocessing import Process

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

from export import add_df_cols

from export_drift_rate import get_noise_df
use_bias_correction = True

max_slot_durs = list(range(10, 201, 4))
#max_slot_durs = list(range(74, 90+1, 4))
#max_slot_durs = list(range(10, 26, 4))

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

def gen_processes():
    for logfile in logfiles:
        for dur in max_slot_durs:
            p = Process(target=get_noise_df, args=(logfile, dur))
            p.daemon = True
            yield p


if __name__ == "__main__":
    import itertools
    config = load_env_config()

    print("V 2")

    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']

    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])
        print("CACHED INITIALIZDED")
    else:
        print("CACHED NOT INITIALIZDED")
        exit()

    proc_gen = gen_processes()

    MAX_PARALLEL = 40

    while True:
        processes = list(itertools.islice(proc_gen, MAX_PARALLEL))

        if len(processes) == 0:
            break

        for p in processes:
            p.start()

        for p in processes:
            p.join()
        print("Step done!")

    print("All done")