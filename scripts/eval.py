import os
import progressbar
import numpy as np
import json

import logs
import utility

from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, gen_delay_estimates_from_testbed_run
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd


import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached, init_cache, load_env_config, set_global_cache_prefix_by_config
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

import pandas as pd

METHOD_PREFIX = 'export_'

CONFIDENCE_FILL_COLOR = '0.8'
PERCENTILES_FILL_COLOR = '0.5'
COLOR_MAP = 'tab10'

c_in_air = 299702547.236


PROTOCOL_NAME = "X"

runs = {
        'trento_a': 'job_fixed',
        'trento_b': None,
        'lille': None
}

src_devs = {
        'trento_a': 'dwm1001.1',  # [(6,3)],
        'trento_b': 'dwm1001.160',
        'lille': 'dwm1001-1'  # [(11,3), (10,3), (7,1), (5,0)],
}

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





def export_testbed_variance(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([lille, trento_a, trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['estimated_tof']
            meas_df = meas_df[['pair', 'estimated_m', 'dist']]

            ma = np.zeros((len(t.devs), len(t.devs)))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if b > a:
                        e = res.loc['{}-{}'.format(a, b), ('estimated_m', 'std')]*100   # in cm
                        ma[a, b] = e
                        ma[b, a] = e
            return ma

        ma = np.array(cached(('meas_var', t.name, runs[t.name], src_devs[t.name], 11), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_ma_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

def export_testbed_variance_calculated_tof(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([lille, trento_a, trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['calculated_tof']
            meas_df = meas_df[['pair', 'estimated_m', 'dist']]

            ma = np.zeros((len(t.devs), len(t.devs)))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if b > a:
                        e = res.loc['{}-{}'.format(a, b), ('estimated_m', 'std')]*100   # in cm
                        ma[a, b] = e
                        ma[b, a] = e
            return ma

        ma = np.array(cached(('meas_var', t.name, runs[t.name], src_devs[t.name], 10), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_ma_calculated_tof_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)
        plt.close()

def export_testbed_variance_calculated_tof_ci(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([lille, trento_a, trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['calculated_tof_ci']
            meas_df = meas_df[['pair', 'estimated_m', 'dist']]

            ma = np.zeros((len(t.devs), len(t.devs)))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if b > a:
                        e = res.loc['{}-{}'.format(a, b), ('estimated_m', 'std')]*100   # in cm
                        ma[a, b] = e
                        ma[b, a] = e
            return ma

        ma = np.array(cached(('var_ma_calculated_tof_ci_', t.name, runs[t.name], src_devs[t.name], 12), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_ma_calculated_tof_ci_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

def export_testbed_variance_calculated_tof_ci_avg(config, export_dir):

    std_upper_lim = 10.0

    for x in [1, 2]:
        num_ci_avg = x

        for (c, t) in enumerate([lille, trento_a, trento_b]):

            def proc():
                meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name], num_ci_drift_avg=num_ci_avg))
                meas_df['estimated_m'] = meas_df['calculated_tof_ci_avg']
                meas_df = meas_df[['pair', 'estimated_m', 'dist']]

                ma = np.zeros((len(t.devs), len(t.devs)))
                res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

                for a in range(len(t.devs)):
                    for b in range(len(t.devs)):
                        if b > a:
                            e = res.loc['{}-{}'.format(a, b), ('estimated_m', 'std')]*100   # in cm
                            ma[a, b] = e
                            ma[b, a] = e
                return ma

            ma = np.array(cached(('export_testbed_variance_calculated_tof_ci_avg', t.name, runs[t.name], src_devs[t.name], 12, num_ci_avg), proc))
            print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

            plt.clf()
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

            for i in range(ma.shape[0]):
                for j in range(ma.shape[1]):
                    if i != j:
                        e = ma[i, j]
                        if e > 100:
                            e = int(ma[i, j])
                        else:
                            e = round(ma[i, j], 1)

                        if e >= std_upper_lim:
                            s = r"\underline{" + str(e) + "}"
                        else:
                            s = str(e)
                        ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

            ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
            ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
            fig.set_size_inches(4.75, 4.75)
            plt.tight_layout()

            plt.savefig("{}/var_ma_calculated_tof_ci_avg_{}_{}.pdf".format(export_dir, num_ci_avg, t.name), bbox_inches='tight', pad_inches=0)

            plt.close()

def export_testbed_variance_from_device(config, export_dir):

    std_upper_lim = 100.0

    for (c, t) in enumerate([trento_b]):

        def proc():
            est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

            round = est_df['round'].max()
            est_df = est_df[(est_df['round'] == round)]


            ma = np.zeros((len(t.devs), len(t.devs)))

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if b > a:
                        res = est_df.loc[est_df['pair'] == '{}-{}'.format(a, b), "var_measurement"]

                        e = 0.0
                        if len(res) > 0:
                            var = (res.to_numpy())[0]
                            if var is not None:
                                e = np.sqrt(var)*100

                        ma[a, b] = e
                        ma[b, a] = e
            return ma

        ma = np.array(cached(('meas_var_device', t.name, runs[t.name], src_devs[t.name], 9), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+1))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+1))
        fig.set_size_inches(6.0, 6.0)
        plt.tight_layout()

        plt.savefig("{}/var_ma_{}_from_device.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

def export_testbed_tdoa_variance(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['estimated_tdoa']
            meas_df = meas_df[['pair', 'estimated_m', 'tdoa']]

            print(meas_df)

            ma = np.zeros((len(t.devs)-1, len(t.devs)-1))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs)-1):
                for b in range(len(t.devs)-1):
                    if b > a:
                        e = res.loc['{}-{}'.format(a+1, b+1), ('estimated_m', 'std')]*100   # in cm
                        ma[a-1, b-1] = e
                        ma[b-1, a-1] = e
            return ma

        ma = np.array(cached(('export_testbed_tdoa_variance', t.name, runs[t.name], src_devs[t.name], 16), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+2))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+2))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_tdoa_ma_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

        #plt.show()

def export_testbed_tdoa_calculated_tdoa_variance(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['calculated_tdoa']
            meas_df = meas_df[['pair', 'estimated_m', 'tdoa']]

            print(meas_df)

            ma = np.zeros((len(t.devs)-1, len(t.devs)-1))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs)-1):
                for b in range(len(t.devs)-1):
                    if b > a:
                        e = res.loc['{}-{}'.format(a+1, b+1), ('estimated_m', 'std')]*100   # in cm
                        ma[a-1, b-1] = e
                        ma[b-1, a-1] = e
            return ma

        ma = np.array(cached(('export_testbed_tdoa_variance_calculated_tdoa', t.name, runs[t.name], src_devs[t.name], 16), proc))
        print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x+2))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x+2))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_tdoa_ma_calculated_tdoa_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

def export_testbed_tdoa_calculated_tdoa_ci_variance(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([trento_b]):

        def proc():
            meas_df = pd.DataFrame.from_records(
                gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))
            meas_df['estimated_m'] = meas_df['calculated_tdoa_ci']
            meas_df = meas_df[['pair', 'estimated_m', 'tdoa']]

            print(meas_df)

            ma = np.zeros((len(t.devs) - 1, len(t.devs) - 1))
            res = meas_df.groupby('pair').aggregate(func=['mean', 'std'])

            for a in range(len(t.devs) - 1):
                for b in range(len(t.devs) - 1):
                    if b > a:
                        e = res.loc['{}-{}'.format(a + 1, b + 1), ('estimated_m', 'std')] * 100  # in cm
                        ma[a - 1, b - 1] = e
                        ma[b - 1, a - 1] = e
            return ma

        ma = np.array(
            cached(('export_testbed_tdoa_variance_calculated_tdoa_ci', t.name, runs[t.name], src_devs[t.name], 16),
                   proc))
        print(t.name, "mean std", ma.mean(), "median", np.median(ma), "max", np.max(ma), "90% quantile",
              np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma, vmin=0.0, vmax=std_upper_lim, norm='asinh')

        for i in range(ma.shape[0]):
            for j in range(ma.shape[1]):
                if i != j:
                    e = ma[i, j]
                    if e > 100:
                        e = int(ma[i, j])
                    else:
                        e = round(ma[i, j], 1)

                    if e >= std_upper_lim:
                        s = r"\underline{" + str(e) + "}"
                    else:
                        s = str(e)
                    ax.text(x=j, y=i, s=s, va='center', ha='center', usetex=True)

        ax.xaxis.set_major_formatter(lambda x, pos: int(x + 2))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x + 2))
        fig.set_size_inches(4.75, 4.75)
        plt.tight_layout()

        plt.savefig("{}/var_tdoa_ma_calculated_tdoa_ci_{}.pdf".format(export_dir, t.name), bbox_inches='tight',
                    pad_inches=0)

        plt.close()



def export_testbed_layouts(config, export_dir):

    # we draw every layout

    # TODO: Extract them from the actual data!
    high_variance_connections = {
        'trento_a': [],#[(6,3)],
        'trento_b': [],
        'lille': [] #[(11,3), (10,3), (7,1), (5,0)],
    }

    always_drawn ={
        'lille': [],
        'trento_a': [],
        'trento_b': []
    }

    limits = {
        'lille': [-0.5, 10.0, 29, 22.5],
        'trento_a': [71, 81, -1.25, 8.5],
        'trento_b': [118.5, 131.5, -1, 8.0]
    }


    for (c,t) in enumerate([lille, trento_a, trento_b]):
        ys = []
        xs = []
        ns = []

        for (i,k) in enumerate(t.devs):
            pos = t.dev_positions[k]
            xs.append(pos[0])
            ys.append(pos[1])
            n = i+1 #k.replace("dwm1001.", "").replace("dwm1001-", "")
            ns.append(n)

        plt.clf()



        fig, ax = plt.subplots()

        t.draw_layout(plt)

        ax.scatter(xs, ys, color='C'+str(c), marker='o')
        ax.set_aspect('equal', adjustable='box')


        for (a,b) in high_variance_connections[t.name]:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)

        for a in always_drawn[t.name]:
            for b in range(len(t.devs)):
                if a > a:
                    plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)


        for i, txt in enumerate(ns):
            ax.annotate(txt, (xs[i], ys[i]), xytext=(-3, 0), textcoords='offset points', va='center', ha='right')

        ax.set_axisbelow(True)
        ax.set_xlabel("Position X [m]")
        ax.set_ylabel("Position Y [m]")

        if t.name == 'lille':
            ax.invert_yaxis()

        plt.axis(limits[t.name])

        fig.set_size_inches(4.0, 3.5)
        plt.tight_layout()

        #plt.show()
        plt.savefig("{}/layout_{}.pdf".format(export_dir, t.name),bbox_inches = 'tight', pad_inches = 0, dpi=600)

        plt.close()



def export_overall_rmse_reduction(config, export_dir):


    # we extract values for:
    # uncalibrated, factory, calibrated_unfiltered, calibrated_filtered

    errs = {}
    ts = [trento_a, trento_b, lille]

    for (c, t) in enumerate(ts):
        def proc():
            est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

            round = est_df['round'].max()



            est_df = est_df[(est_df['round'] == round) & (est_df['mean_measurement'].notna())]
            est_df['err_uncalibrated'] = est_df['est_distance_uncalibrated'] - est_df['dist']
            est_df['err_factory'] = est_df['est_distance_factory'] - est_df['dist']
            est_df['err_calibrated'] = est_df['est_distance_calibrated'] - est_df['dist']
            est_df['err_calibrated_filtered'] = est_df['est_distance_calibrated_filtered'] - est_df['dist']

            #est_df['std'] = est_df['var_measurement'].apply(np.sqrt) * 100.0

            return {
                'err_uncalibrated': est_df['err_uncalibrated'].to_numpy(),
                'err_factory': est_df['err_factory'].to_numpy(),
                'err_calibrated': est_df['err_calibrated'].to_numpy()
                #'err_calibrated_filtered': est_df['err_calibrated_filtered'].to_numpy()
            }

            # est_df['abs_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.abs)
            # est_df['abs_err_factory'] = est_df['err_factory'].apply(np.abs)
            # est_df['abs_err_calibrated'] = est_df['err_calibrated'].apply(np.abs)
            #
            # est_df['squared_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.square)
            # est_df['squared_err_factory'] = est_df['err_factory'].apply(np.square)
            # est_df['squared_err_calibrated'] = est_df['err_calibrated'].apply(np.square)
            #
            # # we determine the mean squared error of all pairs
            #
            # est_df = est_df[['pair', 'squared_err_uncalibrated', 'squared_err_factory', 'squared_err_calibrated']]
            # res = est_df.aggregate(func=['mean'])
            #
            # return {
            #     'rmse_uncalibrated': np.sqrt(res['squared_err_uncalibrated'][0]),
            #     'rmse_factory': np.sqrt(res['squared_err_factory'][0]),
            #     'rmse_calibrated': np.sqrt(res['squared_err_calibrated'][0])
            # }

        data = cached(('overall', t.name, runs[t.name], src_devs[t.name], 17), proc)
        for k in data:
            if k not in errs:
                errs[k] = []
            errs[k].append(data[k]*100)

    def rmse(xs):
        return np.sqrt(np.mean(np.square(np.array(xs)))), 0


    stds = {}

    for k in errs:
        stds[k] = []
        for (i, xs) in enumerate(errs[k]):
            e, sd = rmse(xs)
            errs[k][i] = e*100
            stds[k].append(sd*100)

    scenarios = ["CLOVES-7", "CLOVES-14", "IoT-LAB-14"]

    x = np.arange(len(scenarios))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0


    plt.clf()
    fig, ax = plt.subplots(layout='constrained')

    labels = {
        'err_uncalibrated': "Uncalibrated",
        'err_factory': "Factory",
        'err_calibrated': PROTOCOL_NAME
    }

    for attribute, measurement in errs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=labels[attribute])
        ax.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    ax.set_ylabel('RMSE [cm]')
    ax.set_xlabel('Scenario')
    ax.set_xticks(x + width, scenarios)
    ax.legend()
    plt.gca().yaxis.grid(True, color='lightgray', linestyle='dashed')
    ax.set_ylim(0, 20)

    plt.tight_layout()

    plt.savefig("{}/rmse_comparison.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)


def export_overall_mae_reduction(config, export_dir):


    # we extract values for:
    # uncalibrated, factory, calibrated_unfiltered, calibrated_filtered

    errs = {}
    ts = [trento_a, trento_b, lille]

    for (c, t) in enumerate(ts):
        def proc():
            est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

            round = est_df['round'].max()


            est_df = est_df[(est_df['round'] == round) & (est_df['mean_measurement'].notna())]
            est_df['err_uncalibrated'] = est_df['est_distance_uncalibrated'] - est_df['dist']
            est_df['err_factory'] = est_df['est_distance_factory'] - est_df['dist']
            est_df['err_calibrated'] = est_df['est_distance_calibrated'] - est_df['dist']
            est_df['err_calibrated_filtered'] = est_df['est_distance_calibrated_filtered'] - est_df['dist']

            #est_df['std'] = est_df['var_measurement'].apply(np.sqrt) * 100.0

            return {
                'err_uncalibrated': est_df['err_uncalibrated'].to_numpy(),
                'err_factory': est_df['err_factory'].to_numpy(),
                'err_calibrated': est_df['err_calibrated'].to_numpy(),
                'err_calibrated_filtered': est_df['err_calibrated_filtered'].to_numpy()
            }

            # est_df['abs_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.abs)
            # est_df['abs_err_factory'] = est_df['err_factory'].apply(np.abs)
            # est_df['abs_err_calibrated'] = est_df['err_calibrated'].apply(np.abs)
            #
            # est_df['squared_err_uncalibrated'] = est_df['err_uncalibrated'].apply(np.square)
            # est_df['squared_err_factory'] = est_df['err_factory'].apply(np.square)
            # est_df['squared_err_calibrated'] = est_df['err_calibrated'].apply(np.square)
            #
            # # we determine the mean squared error of all pairs
            #
            # est_df = est_df[['pair', 'squared_err_uncalibrated', 'squared_err_factory', 'squared_err_calibrated']]
            # res = est_df.aggregate(func=['mean'])
            #
            # return {
            #     'rmse_uncalibrated': np.sqrt(res['squared_err_uncalibrated'][0]),
            #     'rmse_factory': np.sqrt(res['squared_err_factory'][0]),
            #     'rmse_calibrated': np.sqrt(res['squared_err_calibrated'][0])
            # }

        data = cached(('overall_mae', t.name, runs[t.name], src_devs[t.name], 17), proc)
        for k in data:
            if k not in errs:
                errs[k] = []
            errs[k].append(data[k]*100)

    def ae(xs):
        return np.abs(np.array(xs)).mean(), np.abs(np.array(xs)).std()


    stds = {}

    for k in errs:
        stds[k] = []
        for (i, xs) in enumerate(errs[k]):
            e, sd = ae(xs)
            errs[k][i] = e*100
            stds[k].append(sd*100)

    scenarios = ["CLOVES-7", "CLOVES-14", "IoT-LAB-14"]

    x = np.arange(len(scenarios))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    plt.clf()
    fig, ax = plt.subplots(layout='constrained')

    labels = {
        'err_uncalibrated': "Uncalibrated",
        'err_factory': "Factory",
        'err_calibrated': PROTOCOL_NAME,
        #'err_calibrated_filtered': "Calibrated Filtered",
    }

    for attribute, measurement in errs.items():
        if attribute in labels:
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=labels[attribute], yerr=stds[attribute])

            ax.bar_label(rects, padding=0, fontsize=9, label_type='edge', labels=["{:.1f}\n[{:.1f}]".format(round(measurement[i],1), round(stds[attribute][i],1)) for i in range(len(measurement))])

            # for (i, r) in enumerate(rects):
            #     print(r)
            #     ax.text(r.xy[0] + r._width / 2, -1.5, '{:.2f}'.format(r._height), fontsize=10, color='black',
            #             ha='center', va='bottom')


            multiplier += 1

    ax.set_ylabel('MAE [cm]')
    ax.set_xlabel('Scenario')
    ax.set_xticks(x + width, scenarios)
    ax.legend(loc='upper center')
    ax.set_ylim(None, 30)

    plt.gca().yaxis.grid(True, color='lightgray', linestyle='dashed')
    fig.set_size_inches(5.5, 4.5)

    plt.tight_layout()
    plt.savefig("{}/mae_comparison.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)


def export_trento_a_pairs(config, export_dir):

    # we extract values for:
    # uncalibrated, factory, calibrated_unfiltered, calibrated_filtered

    errs = {}
    ts = [trento_a]

    for (c, t) in enumerate(ts):
        est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

        round = est_df['round'].max()

        est_df = est_df[(est_df['round'] == round) & (est_df['mean_measurement'].notna())]

        est_df['Uncalibrated'] = (est_df['est_distance_uncalibrated'] - est_df['dist'])*100
        est_df['Factory'] = (est_df['est_distance_factory'] - est_df['dist'])*100
        est_df[PROTOCOL_NAME] = (est_df['est_distance_calibrated'] - est_df['dist'])*100

        est_df = est_df.sort_values(by='Uncalibrated')

        def rename_pairs(x=None):
            return '{}-{}'.format(x['initiator']+1, x['responder']+1)

        est_df['pair'] = est_df.apply(rename_pairs, axis=1)

        # df.plot.bar(x='pair',y=['dist', 'est_distance_uncalibrated', 'est_distance_factory', 'est_distance_calibrated'])
        est_df.plot.bar(x='pair', y=['Uncalibrated', 'Factory', PROTOCOL_NAME], width=0.75)

        plt.legend()
        plt.xlabel("Pair")
        plt.ylabel("Error [cm]")

        plt.gcf().set_size_inches(5.0, 4.0)
        plt.tight_layout()

        plt.gca().yaxis.grid(True, color='lightgray', linestyle='dashed')

        plt.savefig("{}/error_sorted.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
        plt.close()



import testbed_to_c_vals

def export_filtered_mae_reduction(config, export_dir):


    # we extract values for:
    # uncalibrated, factory, calibrated_unfiltered, calibrated_filtered

    errs = {}
    ts = [trento_a, lille]

    # their actual ids not their indices
    ignored_pair = {
        'trento_a': [(6,3)],
        'trento_b': [],
        'lille': [(7,1), (4, 2)],
    }

    for (c, t) in enumerate(ts):
        def proc():
            est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

            est_df = est_df[(est_df['round'] == est_df['round'].max()) & (est_df['mean_measurement'].notna())]


            # we manually calculate the errors for the filtered case and exclude pair 7-4 (or 6,3)
            M = testbed_to_c_vals.create_inference_matrix(len(t.devs), ignored_pairs=ignored_pair[t.name])

            num_pairs = round(len(t.devs) * (len(t.devs) - 1) / 2)
            actual = np.zeros(shape=num_pairs)
            measured = np.zeros(shape=num_pairs)

            for (a, da) in enumerate(t.devs):
                for (b, db) in enumerate(t.devs):
                    if b > a:
                        actual[pair_index(a, b)] = get_dist(t.dev_positions[da], t.dev_positions[db])
                        measured[pair_index(a, b)] = est_df[est_df['pair']== '{}-{}'.format(a,b)]['mean_measurement']

            diffs = measured - actual
            delays_in_m = np.matmul(M, 2 * diffs)

            filtered_res_in_m = np.zeros(shape=round(len(t.devs) * (len(t.devs) - 1) / 2))

            for (a, da) in enumerate(t.devs):
                for (b, db) in enumerate(t.devs):
                    if b > a:
                        filtered_res_in_m[pair_index(a, b)] = measured[pair_index(a, b)]-0.5*delays_in_m[a]-0.5*delays_in_m[b]

            est_df['err_calibrated_filtered'] = filtered_res_in_m - actual
            est_df['err_uncalibrated'] = est_df['est_distance_uncalibrated'] - est_df['dist']
            est_df['err_factory'] = est_df['est_distance_factory'] - est_df['dist']
            est_df['err_calibrated'] = est_df['est_distance_calibrated'] - est_df['dist']

            for (a,b) in ignored_pair[t.name]:
                est_df = est_df[est_df['pair'] != '{}-{}'.format(a,b)]

            return {
                'err_uncalibrated': est_df['err_uncalibrated'].to_numpy(),
                'err_factory': est_df['err_factory'].to_numpy(),
                'err_calibrated': est_df['err_calibrated'].to_numpy(),
                'err_calibrated_filtered': est_df['err_calibrated_filtered'].to_numpy()
            }


        data = cached(('export_filtered_mae_reduction', t.name, runs[t.name], src_devs[t.name], hash(json.dumps(ignored_pair)), 1), proc)
        print(data)
        for k in data:
            if k not in errs:
                errs[k] = []
            errs[k].append(data[k]*100)

    def ae(xs):
        return np.abs(np.array(xs)).mean(), np.abs(np.array(xs)).std()

    stds = {}

    for k in errs:
        stds[k] = []
        for (i, xs) in enumerate(errs[k]):
            e, sd = ae(xs)
            errs[k][i] = e*100
            stds[k].append(sd*100)

    scenarios = ["CLOVES-7", "IoT-Lab-14"]

    x = np.arange(len(scenarios))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0.0


    plt.clf()
    fig, ax = plt.subplots(layout='constrained')

    labels = {
        'err_uncalibrated': "Uncalibrated",
        'err_factory': "Factory",
        'err_calibrated': PROTOCOL_NAME +" Unfiltered",
        'err_calibrated_filtered': PROTOCOL_NAME + " Filtered",
    }

    colors = ['C0', 'C1', 'C2', 'C3']
    color = 0
    for attribute, measurement in errs.items():
        if attribute in labels:
            offset = width * multiplier
            rects = ax.bar(x + offset - width/2, measurement, width, label=labels[attribute], yerr=stds[attribute], color=colors[color])

            ax.bar_label(rects, padding=0, fontsize=9, label_type='edge',
                         labels=["{:.1f}\n[{:.1f}]".format(round(measurement[i], 1), round(stds[attribute][i], 1)) for i
                                 in range(len(measurement))])

            multiplier += 1
            color+=1

    ax.set_ylabel('MAE over Pairs [cm]')
    ax.set_xlabel('Scenario')
    ax.set_xticks(x + width, scenarios)
    ax.legend(loc="upper left")
    plt.gca().yaxis.grid(True, color='lightgray', linestyle='dashed')

    ax.set_ylim(None, 30)

    fig.set_size_inches(5.5, 4.5)

    plt.tight_layout()
    plt.savefig("{}/mae_reduction_filtered.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)



def export_scatter_graph_trento_a(config, export_dir):

    t = trento_a
    src_dev = trento_a.devs[0]

    meas_df = pd.DataFrame.from_records(gen_measurements_from_testbed_run(t, runs[t.name], src_dev=src_dev, include_dummy=True))
    # we first plot the number of measurements for all pairs
    meas_df = meas_df[(meas_df['device'] == src_dev)]

    meas_df = meas_df[(meas_df['round'] <= 1200) & (meas_df['round'] > 200)]

    meas_df['round'] = meas_df['round'] - 200

    meas_df['offset'] = (meas_df['estimated_tof'] - meas_df['dist'])*100

    initiator = 6
    responder_a = 2
    responder_b = 3

    df = meas_df
    df = df[(df['initiator'] == initiator) & ((df['responder'] == responder_a))]

    df_b = meas_df[(meas_df['initiator'] == initiator) & ((meas_df['responder'] == responder_b))]

    ax = df_b.plot(kind='scatter', x='round', y='offset', color='C0', label='{}-{}'.format( initiator+1, responder_b+1), alpha=0.5, figsize=(5, 4), edgecolors='none')
    ax = df.plot(ax=ax, kind='scatter', x='round', y='offset', color='C1', label='{}-{}'.format(initiator+1, responder_a+1), alpha=0.5, edgecolors='none')

    #plt.axhline(y=get_dist(dev_positions[d0], dev_positions[devs[other]]), color='b', linestyle='-')

    #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_single', color='b', label='C (32 bit)')
    #ax = df.plot(ax=ax, kind='scatter', x='round', y='calculated_tof_int_10', color='b', label='Python (Integer)')

    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(loc='upper left')

    ax.set_ylabel('Offset [cm]')
    ax.set_xlabel('Round')

    plt.gcf().set_size_inches(5.0, 4.5)
    plt.tight_layout()

    #plt.title("Scatter {}-{}".format(i, other))
    plt.savefig("{}/measurements_scatter_trento_a.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)
    plt.close()


def export_tdoa_simulation_drift_performance(config, export_dir):

    from sim_tdoa import sim

    xs = [0.0, 1.0e-06 , 1.0e-05, 1.0e-04, 1.0e-03]
    num_repetitions = 100000

    data_rows = []

    for x in xs:


        res, _ = sim(
            num_exchanges=num_repetitions,
            resp_delay_s=(1.0, 0.000001),
            node_drift_std=x,
            rx_noise_std=1.0e-09,
            tx_delay_mean=0.0,
            tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
        )

        tof_std = (res['est_tof_a']).std() * c_in_air
        tdoa_std = (res['est_tof_a_ds']).std() * c_in_air

        data_rows.append(
            {
                'node_drift_std': x,
                'tof_std': 1.0e09*tof_std/c_in_air,
                'tdoa_std': 1.0e09*tdoa_std/c_in_air,
            }
        )


    df = pd.DataFrame(data_rows)

    df = df.rename(columns={"tof_std": "ToF SD", "tdoa_std": "TDoA SD"})

    plt.clf()
    ax = df.plot.bar(x='node_drift_std', y=['ToF SD', 'TDoA SD'], width=0.8)


    plt.ylim(0.0, 2.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Node Drift SD")
    ax.set_ylabel("Sample SD [cm]")

    counter = 0
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            height = 0

        ax.text(p.get_x() + p.get_width() / 2., height,
                "{:.2f}".format(height), fontsize=9, color='black', ha='center',
                va='bottom')

        # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
        counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/tdoa_sim_rmse_drifts.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()


def export_tdoa_simulation_rx_noise(config, export_dir):

    from sim_tdoa import sim, c_in_air

    xs = [2.5, 5, 10, 20]
    num_repetitions = 1000000

    data_rows = []

    for x in xs:

        res, _ = sim(
            num_exchanges=num_repetitions,
            resp_delay_s=0.001,
            node_drift_std=1.0e-06,
            rx_noise_std=x/100.0,
            tx_delay_mean=0.0,
            tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
        )

        tof_std = (res['est_tof_a']).std() * c_in_air
        tdoa_std = (res['est_tof_a_ds']).std() * c_in_air

        data_rows.append(
            {
                'rx_noise_std': x,
                'tof_std': tof_std,
                'tdoa_std': tdoa_std,
            }
        )


    df = pd.DataFrame(data_rows)

    df = df.rename(columns={"tof_std": "ToF SD", "tdoa_std": "TDoA SD"})

    plt.clf()
    ax = df.plot.bar(x='rx_noise_std', y=['ToF SD', 'TDoA SD'], width=0.8)


    plt.ylim(0.0, 1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Reception Noise SD")
    ax.set_ylabel("Sample SD [cm]")

    counter = 0
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            height = 0

        ax.text(p.get_x() + p.get_width() / 2., height,
                "{:.3f}".format(height), fontsize=9, color='black', ha='center',
                va='bottom')

        # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
        counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/tdoa_sim_rmse_rx_noise.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()


def export_tdoa_simulation_response_std(config, export_dir):

    from sim_tdoa import sim

    limit = 2.0
    step = 0.05
    response_delay_exps = np.arange(-limit, limit+step, step)

    xs = response_delay_exps
    num_sim_per_rep = 512
    num_reps = 16
    node_drift_std = 10.0/1000000.0
    mitigate_drift = True
    rx_noise_std = 1.0e-09

    rx_noise_stds = {
        'a-b': rx_noise_std,
        'b-a': rx_noise_std,
        'a-p': rx_noise_std,
        'b-p': rx_noise_std,
    }

    def get_rx_noise(tx, rx):
        if isinstance(rx_noise_stds, dict):
            return rx_noise_stds["{}-{}".format(tx, rx)]
        else:
            return rx_noise_stds

    def calc_predicted_tof_std(delay_b, delay_a):
        a_b_std = get_rx_noise('a', 'b')
        b_a_std = get_rx_noise('b', 'a')

        return np.sqrt(
            (0.5 * b_a_std) ** 2
            + (0.5 * (delay_b / (delay_a + delay_b)) * a_b_std) ** 2
            + (0.5 * (1 - (delay_b / (delay_a + delay_b))) * a_b_std) ** 2
        )

    def calc_predicted_tdoa_std(delay_b, delay_a):
        a_b_std = get_rx_noise('a', 'b')
        b_a_std = get_rx_noise('b', 'a')
        a_p_std = get_rx_noise('a', 'p')
        b_p_std = get_rx_noise('b', 'p')

        comb_delay = delay_a+delay_b

        return np.sqrt(
            (0.5 * b_a_std) ** 2
            + (0.5 * a_b_std * (-1 + (delay_b/comb_delay))) ** 2
            + (0.5 * a_b_std * (-delay_b/comb_delay)) ** 2
            + (0.5 * a_p_std * ((-delay_b/comb_delay)-(delay_b/comb_delay)+2)) ** 2
            + (0.5 * b_p_std * (-2)) ** 2
            + (0.5 * a_p_std * (delay_b/comb_delay+delay_b/comb_delay)) ** 2
        )


    for resp_delay_s in [0.002]:

        def proc():
            data_rows = []
            prediction_rows = []

            for x in xs:
                delay_a = resp_delay_s * np.power(10, x)
                delay_b = resp_delay_s

                for i in range(num_reps):
                    res, _ = sim(
                        num_exchanges=num_sim_per_rep,
                        resp_delay_s=(delay_b, delay_a),
                        node_drift_std=node_drift_std,
                        rx_noise_std=rx_noise_stds,
                        tx_delay_mean=0.0,
                        tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0,
                        mitigate_drift=mitigate_drift
                    )

                    data_rows.append(
                        {
                            'rdr': x,
                            'tof_std': 1.0e09*(res['est_tof_a']).std(),
                            'tof_ds_std': 1.0e09*(res['est_tof_a_ds']).std(),
                            'tdoa_std': 1.0e09*(res['est_tdoa']).std(),
                            'tdoa_ds_std': 1.0e09*(res['est_tdoa_ds']).std(),
                            'tdoa_half_cor_std': 1.0e09*(res['est_tdoa_half_cor']).std(),
                            'tdoa_ds_half_cor_std': 1.0e09*(res['est_tdoa_ds_half_cor']).std(),
                            'tof_mean': 1.0e09 * (res['est_tof_a']).mean(),
                            'tof_ds_mean': 1.0e09 * (res['est_tof_a_ds']).mean(),
                            'tdoa_mean': 1.0e09 * (res['est_tdoa']).mean(),
                            'tdoa_ds_mean': 1.0e09 * (res['est_tdoa_ds']).mean(),
                            'tdoa_half_cor_mean': 1.0e09 * (res['est_tdoa_half_cor']).mean(),
                            'tdoa_ds_half_cor_mean': 1.0e09 * (res['est_tdoa_ds_half_cor']).mean()
                        }
                    )

                prediction_rows.append({
                    'rdr': x,
                    'predicted_tof_std': 1.0e09 * calc_predicted_tof_std(delay_b, delay_a),
                    'predicted_tdoa_std': 1.0e09 * calc_predicted_tdoa_std(delay_b, delay_a)
                })
            return data_rows, prediction_rows

        data_rows, prediction_rows = cached( ('export_tdoa_simulation_response_std_new', limit, step, 14, num_sim_per_rep, num_reps, resp_delay_s, node_drift_std, mitigate_drift), proc)

        data_df = pd.DataFrame(data_rows)
        pred_df = pd.DataFrame(prediction_rows)

        data_df = data_df.rename(columns={
            "tof_std": "Simulated ToF SD",
            "tof_ds_std": "ToF DS SD",
            "tdoa_std": "Simulated TDoA SD",
            "tdoa_ds_std": "TDoA DS SD",
            "tdoa_half_cor_std": "TDoA (w/ DC) SD",
            "tdoa_ds_half_cor_std": "TDoA DS (w/ DC) SD",
        })

        pred_df = pred_df.rename(columns={
            "predicted_tof_std": "Analytical ToF SD",
            "predicted_tdoa_std": "Analytical TDoA SD"
        })

        plt.clf()
        ax = pred_df.plot.line(x='rdr', y=[
            'Analytical ToF SD',
            'Analytical TDoA SD'
        ], alpha=0.5, color=['C4', 'C2'])

        data_df.plot.scatter(x='rdr', y='Simulated ToF SD', ax=ax, c='C4', s=0.5, label='Simulated ToF SD')
        data_df.plot.scatter(x='rdr', y='Simulated TDoA SD', ax=ax, c='C2', s=0.5, label='Simulated TDoA SD')


        ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

        #plt.axhline(y=np.sqrt(0.5), color='C0', linestyle='dotted', label = "Analytical ToF SD")
        #plt.axhline(y=np.sqrt(2.5), color='C1', linestyle='dotted', label = "Analytical TDoA SD")



        #plt.ylim(0.2, 1.8)
        #plt.ylim(0.2, 15)

        ax.set_axisbelow(True)
        ax.set_xlabel("Delay Ratio $\\frac{D_A}{D_B}$")
        ax.set_ylabel("Sample SD [ns]")

        from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))


        # counter = 0
        # for p in ax.patches:
        #     height = p.get_height()
        #     if np.isnan(height):
        #         height = 0
        #
        #     ax.text(p.get_x() + p.get_width() / 2., height,
        #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
        #             va='bottom')
        #
        #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
        #     counter += 1


        plt.grid(color='lightgray', linestyle='dashed')

        plt.legend(loc='lower left')
        plt.gcf().set_size_inches(6.0, 4.5)

        ticks = list(ax.get_yticks())
        labels = list(ax.get_yticklabels())

        #ticks.append(1.0e09 * np.sqrt((0.5*get_rx_noise('a', 'b'))**2 + (0.5*get_rx_noise('b', 'a'))**2))
        #labels.append(r'$\sqrt{0.5^2 \sigma_{BA}^2 + 0.5^2 \sigma_{AB}^2}$')

        ticks.append(np.sqrt(0.5))
        ticks.append(np.sqrt(2.5))

        labels.append(r'$\sqrt{0.5}\sigma$')
        labels.append(r'$\sqrt{2.5}\sigma$')

        ticks.append(np.sqrt(0.375))
        ticks.append(np.sqrt(1.875))
        labels.append(r'$\sqrt{0.375}\sigma$')
        labels.append(r'$\sqrt{1.875}\sigma$')

        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

        print(ticks)
        print(labels)


        plt.tight_layout()

        plt.savefig("{}/tdoa_sim_rmse_reponse_delay_ratio_{}.pdf".format(export_dir, round(resp_delay_s*1000)), bbox_inches = 'tight', pad_inches = 0)
        #plt.show()

        plt.close()


def export_tof_simulation_response_std(config, export_dir):

    from sim_tdoa import sim

    limit = 6.0
    step = 0.1
    response_delay_exps = np.arange(-limit, limit+step, step)

    xs = response_delay_exps
    num_sims = 1000
    num_repetitions = 1
    resp_delay_s = 1.0

    rx_noise_std = 1.0e-09


    def proc():
        data_rows = []
        for x in xs:

            res, _ = sim(
                num_exchanges=num_sims,
                resp_delay_s=(resp_delay_s, resp_delay_s*np.power(10, x)),
                node_drift_std=100.0/1000000.0,
                rx_noise_std=rx_noise_std,
                tx_delay_mean=0.0,
                tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
            )

            data_rows.append(
                {
                    'rdr': x,
                    'tof_mean': 1.0e09*res['est_tof_a'].mean(),
                    'tof_std': 1.0e09*res['est_tof_a'].std(),
                }
            )

        return data_rows

    data_rows = cached(('export_tdoa_simulation_response_mean', limit, step, 10, num_sims, resp_delay_s), proc)

    df = pd.DataFrame(data_rows)

    df = df.rename(columns={"tof_mean": "ToF Mean", "tof_std": "ToF SD"})

    plt.clf()
    ax = df.plot.line(x='rdr', y=['ToF Mean'])

    #ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

    #plt.ylim(0.2, 1.8)

    ax.set_axisbelow(True)
    ax.set_xlabel("Delay Ratio $\\frac{D_A}{D_B}$")
    ax.set_ylabel("Sample SD [ns]")

    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

    #ax.yaxis.set_major_locator(MultipleLocator(1.0))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.2))


    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.legend(ncol=2)
    plt.gcf().set_size_inches(6.0, 4.5)

    ticks = list(ax.get_yticks())
    labels = list(ax.get_yticklabels())

    #ticks.append(np.sqrt(0.5))
    #ticks.append(np.sqrt(2.5))

    #labels.append(r'$\sqrt{0.5}\sigma$')
    #labels.append(r'$\sqrt{2.5}\sigma$')

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)

    #print(ticks)
    #print(labels)


    plt.tight_layout()

    plt.savefig("{}/sim_rmse_reponse_delay_ratio.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()

def export_tdoa_simulation_response_std_scatter(config, export_dir):

    from sim_tdoa import sim

    response_delay_exps = np.arange(-6.0, 6+1, 0.25)

    xs = response_delay_exps
    num_sims = 100
    resp_delay_s = 1.0


    def proc():
        data_rows = []
        for x in xs:

            _, run_data_rows = sim(
                num_exchanges=num_sims,
                resp_delay_s=(resp_delay_s, resp_delay_s * np.power(10, x)),
                node_drift_std=10.0 / 1000000.0,
                rx_noise_std=1.0e-09,
                tx_delay_mean=0.0,
                tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0
            )

            for row in run_data_rows:
                data_rows.append(
                    {
                        'rdr': x,
                        'tof_std': 1.0e09*(row['est_tof_a']-row['real_tof']),
                        'tdoa_std': 1.0e09*(row['est_tdoa']-row['real_tdoa']),
                    }
                )

        return data_rows

    data_rows = cached( ('export_tdoa_simulation_response_std_scatter', hash(json.dumps(list(xs))), 9, num_sims, resp_delay_s, num_sims), proc)
    df = pd.DataFrame(data_rows)
    print(df)

    df = df.rename(columns={"tof_std": "ToF SD", "tdoa_std": "TDoA SD"})

    plt.clf()
    #ax = df.plot.line(x='rdr', y=['ToF SD', 'TDoA SD'])
    ax = df.plot.scatter(x='rdr', y='ToF SD', alpha=0.2)
    #ax = df.plot.scatter(x='rdr', y='TDoA SD')


    plt.ylim(-4.0, 4.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Response Ratio")
    ax.set_ylabel("Sample SD [ns]")

    # counter = 0
    # for p in ax.patches:
    #     height = p.get_height()
    #     if np.isnan(height):
    #         height = 0
    #
    #     ax.text(p.get_x() + p.get_width() / 2., height,
    #             "{:.2f}".format(height), fontsize=9, color='black', ha='center',
    #             va='bottom')
    #
    #     # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')
    #     counter += 1


    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/export_tdoa_simulation_response_std_scatter.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()

def export_loc_sim(config, export_dir):
    import sim_localization_variance

    samples_per_side = 200
    repetitions = 500
    meas_std = 0.05

    def tof_proc():
        return sim_localization_variance.create_matrix(
            sim_localization_variance.sim_single_tof_positioning,
            meas_std=meas_std,
            samples_per_side=samples_per_side,
            repetitions=repetitions
        )

    def tdoa_proc():
        return sim_localization_variance.create_matrix(
            sim_localization_variance.sim_single_tdoa_positioning,
            meas_std=meas_std*np.sqrt(2.5)/np.sqrt(0.5),
            samples_per_side=samples_per_side,
            repetitions=repetitions
        )

    version = 4
    tof_m = cached(('export_tof_loc_sim', meas_std, samples_per_side, repetitions, sim_localization_variance.SIDE_LENGTH, version), tof_proc)
    tdoa_m = cached(('export_tdoa_loc_sim', meas_std, samples_per_side, repetitions, sim_localization_variance.SIDE_LENGTH, version), tdoa_proc)

    exps = {
        "tof": tof_m,
        "tdoa": tdoa_m
    }

    for k in exps:
        plt.clf()
        fig, ax = plt.subplots()
        im = ax.imshow(exps[k], interpolation='none') #, vmin=0.0, vmax=5.0)

        for (x, y) in sim_localization_variance.NODES_POSITIONS:
            plt.plot(
                (y / sim_localization_variance.SIDE_LENGTH) * samples_per_side,
                (x / sim_localization_variance.SIDE_LENGTH) * samples_per_side,
                'wx',
            )

        cbar = ax.figure.colorbar(im, ax=ax, location='right', format=lambda x, _: "{:.2f}".format(x), shrink=0.68, pad=0.025)
        cbar.ax.set_ylabel("Localization RMSE [m]", rotation=-90, va="bottom")

        ax.xaxis.set_major_locator(MultipleLocator(2*round(samples_per_side / sim_localization_variance.SIDE_LENGTH)))
        ax.yaxis.set_major_locator(MultipleLocator(2*round(samples_per_side / sim_localization_variance.SIDE_LENGTH)))
        ax.invert_yaxis()


        ax.xaxis.set_major_formatter(lambda x, pos: round((x/ samples_per_side) * sim_localization_variance.SIDE_LENGTH))
        ax.yaxis.set_major_formatter(lambda x, pos: round((x/ samples_per_side) * sim_localization_variance.SIDE_LENGTH))

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        plt.xlim(0.0, (10.0/sim_localization_variance.SIDE_LENGTH)*samples_per_side)
        plt.ylim(0.0, (10.0/sim_localization_variance.SIDE_LENGTH)*samples_per_side)


        fig.set_size_inches(4.0, 4.0)
        plt.tight_layout()

        #plt.show()
        plt.savefig("{}/export_{}_loc_sim.pdf".format(export_dir, k), bbox_inches = 'tight', pad_inches = 0)
        plt.close()


def get_df(log, tdoa_src_dev_number, use_bias_correction):
    def proc():
        it = logs.extract_tdma_twr(trento_b, log, tdoa_src_dev_number=tdoa_src_dev_number, bias_corrected=use_bias_correction)
        return pd.DataFrame.from_records(it)

    df = utility.cached_dt(('extract_job_tdma', log, tdoa_src_dev_number, use_bias_correction), proc)

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

def extract_active_and_all_passive_dfs(log, filter_pair, filter_passive_listener, use_bias_correction, skip_to_round=0,
                                   up_to_round=None):
    active_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)

    if filter_passive_listener is not None:
        passive_df = get_df(log, tdoa_src_dev_number=filter_passive_listener,
                            use_bias_correction=use_bias_correction)
    else:
        tdoa_dfs = []

        for dev_num in range(len(trento_b.devs)):
            tdoa_dfs.append(get_df(log, tdoa_src_dev_number=dev_num, use_bias_correction=use_bias_correction))
        passive_df = pd.concat(tdoa_dfs)

    if skip_to_round is not None:
        active_df = active_df[active_df['round'] >= skip_to_round]
        passive_df = passive_df[passive_df['round'] >= skip_to_round]

    if up_to_round is not None:
        active_df = active_df[active_df['round'] <= up_to_round]
        passive_df = passive_df[passive_df['round'] <= up_to_round]

    if filter_pair is not None:
        active_df = active_df[active_df['pair'] == filter_pair]
        passive_df = passive_df[passive_df['pair'] == filter_pair]

    if passive_df['tdoa_est_ds'].count() != 0:
        passive_df['tdoa_est_ds_err'] = passive_df['tdoa_est_ds'] - passive_df['tdoa']
        passive_df['tdoa_est_ss_init_err'] = passive_df['tdoa_est_ss_init'] - passive_df['tdoa']
        passive_df['tdoa_est_ss_final_err'] = passive_df['tdoa_est_ss_final'] - passive_df['tdoa']
        passive_df['tdoa_est_ss_both_err'] = passive_df['tdoa_est_ss_both'] - passive_df['tdoa']
        passive_df['tdoa_est_mixed_err'] = passive_df['tdoa_est_mixed'] - passive_df['tdoa']

    return active_df, passive_df


def compute_means_and_stds(log, filter_pair, filter_passive_listener, use_bias_correction, skip_to_round=0,
                           up_to_round=None):
    active_df, passive_df = extract_active_and_all_passive_dfs(log, filter_pair, filter_passive_listener,
                                                           use_bias_correction, skip_to_round=skip_to_round,
                                                           up_to_round=up_to_round)
    active_agg = active_df.agg(
        twr_count=pd.NamedAgg(column='twr_tof_ds', aggfunc="count"),
        dist=pd.NamedAgg(column='dist', aggfunc="max"),
        twr_tof_ds_err_mean=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="mean"),
        twr_tof_ds_mae=pd.NamedAgg(column='twr_tof_ds_err', aggfunc=lambda x: x.abs().mean()),
        twr_tof_ds_err_std=pd.NamedAgg(column='twr_tof_ds_err', aggfunc="std"),
        twr_tof_ss_err_mean=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="mean"),
        twr_tof_ss_mae=pd.NamedAgg(column='twr_tof_ss_err', aggfunc=lambda x: x.abs().mean()),
        twr_tof_ss_err_std=pd.NamedAgg(column='twr_tof_ss_err', aggfunc="std"),
        twr_tof_ss_reverse_err_mean=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="mean"),
        twr_tof_ss_reverse_mae=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc=lambda x: x.abs().mean()),
        twr_tof_ss_reverse_err_std=pd.NamedAgg(column='twr_tof_ss_reverse_err', aggfunc="std"),
    )

    active_agg = active_agg.transpose().agg('max')
    active_dict = active_agg.to_dict()

    #active_agg.to_csv('ds-vs-cfg-pair-{}-passive-{}-active.csv'.format(filter_pair, filter_passive_listener))

    if passive_df['tdoa_est_ds'].count() != 0:

        passive_agg = passive_df.agg(
            tdoa_count=pd.NamedAgg(column='tdoa_est_ds', aggfunc="count"),
            tdoa=pd.NamedAgg(column='tdoa', aggfunc="max"),
            tdoa_est_ds_err_mean=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="mean"),
            tdoa_est_ds_mae=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ds_err_std=pd.NamedAgg(column='tdoa_est_ds_err', aggfunc="std"),
            tdoa_est_ss_init_err_mean=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="mean"),
            tdoa_est_ss_init_mae=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_init_err_std=pd.NamedAgg(column='tdoa_est_ss_init_err', aggfunc="std"),
            tdoa_est_ss_both_err_mean=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="mean"),
            tdoa_est_ss_both_mae=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_both_err_std=pd.NamedAgg(column='tdoa_est_ss_both_err', aggfunc="std"),
            tdoa_est_ss_final_err_mean=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="mean"),
            tdoa_est_ss_final_mae=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_ss_final_err_std=pd.NamedAgg(column='tdoa_est_ss_final_err', aggfunc="std"),
            tdoa_est_mixed_err_mean=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="mean"),
            tdoa_est_mixed_mae=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc=lambda x: x.abs().mean()),
            tdoa_est_mixed_err_std=pd.NamedAgg(column='tdoa_est_mixed_err', aggfunc="std")
        )
        #passive_agg.to_csv('ds-vs-cfg-pair-{}-passive-{}-passive.csv'.format(filter_pair, filter_passive_listener))
        passive_agg = passive_agg.transpose().agg('max')
        passive_dict = passive_agg.to_dict()

        for k in passive_dict:
            active_dict[k] = passive_dict[k]

    return active_dict


def compute_all_agg_means_and_stds(log, use_bias_correction = True, skip_to_round = 0, up_to_round = None):
    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)
    passive_listeners = list(range(len(trento_b.devs)))
    pairs = list(twr_df['pair'].unique())

    rows = []
    for filter_passive_listener in passive_listeners:
        for filter_pair in pairs:
            res = compute_means_and_stds(log, filter_pair, filter_passive_listener, use_bias_correction, skip_to_round=skip_to_round, up_to_round=up_to_round)
            res['_filter_pair'] = filter_pair
            res['_filter_passive_listener'] = filter_passive_listener
            rows.append(res)
    return pd.DataFrame.from_records(rows)


def cached_compute_all_agg_means_and_stds(log, use_bias_correction = True, skip_to_round = 0, up_to_round = None):
    def proc():
        return compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round, up_to_round=up_to_round)
    return utility.cached_dt(('cached_compute_all_means_and_stds', log, use_bias_correction,skip_to_round, up_to_round, 1), proc)


def export_twr_scatter(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True

    fig, ax = plt.subplots()

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')


    pair = '9-7'

    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)
    twr_df = twr_df[twr_df['pair'] == pair]
    twr_df.plot.scatter('round', 'twr_tof_ds_err', ax=ax, color='black', label="TWR")

    twr_df = get_df(log, tdoa_src_dev_number=4, use_bias_correction=use_bias_correction)
    twr_df = twr_df[twr_df['pair'] == pair]
    twr_df['tdoa_est_ds_err'] = twr_df['tdoa_est_ds'] - twr_df['tdoa']
    twr_df.plot.scatter('round', 'tdoa_est_ds_err', ax=ax, color='blue', label="TDoA 4")

    twr_df = get_df(log, tdoa_src_dev_number=11, use_bias_correction=use_bias_correction)
    twr_df = twr_df[twr_df['pair'] == pair]
    twr_df['tdoa_est_ds_err'] = twr_df['tdoa_est_ds'] - twr_df['tdoa']
    twr_df.plot.scatter('round', 'tdoa_est_ds_err', ax=ax, color='red', label="TDoA 11")

    q_low = twr_df["tdoa_est_ds_err"].quantile(0.01)
    q_hi = twr_df["tdoa_est_ds_err"].quantile(0.99)
    df_filtered = twr_df[(twr_df["tdoa_est_ds_err"] < q_hi) & (twr_df["tdoa_est_ds_err"] > q_low)]
    df_filtered.plot.scatter('round', 'tdoa_est_ds_err', ax=ax, color='yellow', label="TDoA 11 Filtered")

    plt.show()
    exit()


def export_twr_vs_tdoa_scatter_rssi(config, export_dir):
    log = 'job_tdma_long'

    def proc():
        return pd.DataFrame.from_records(logs.gen_all_rx_events(trento_b, log))

    all_raw_df = utility.cached_dt(('export_twr_vs_tdoa_scatter_rssi_raw', log), proc)
    rssi_values = all_raw_df.groupby(['own_number', 'rx_number']).agg({'rssi': ['mean', 'std', lambda x: x.quantile(0.95)]}).transpose().to_dict()

    def get_rssi_mean_std(r, t):
        return rssi_values[(r,t)][('rssi', 'mean')], rssi_values[(r,t)][('rssi', 'std')]

    use_bias_correction = True
    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=0)

    all_df = all_df[all_df['tdoa_count'].notna()]

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    # gb = all_df.groupby(by='_filter_pair').agg('median') #'.quantile([0, 0.25, 0.5, 0.75, 0.95, 1])
    # all_df = all_df[['ratio']]
    # all_df['ratio'].boxplot()

    fig, ax = plt.subplots()
    all_df.plot.scatter('twr_tof_ds_err_std', 'tdoa_est_ds_err_std', ax=ax)

    def get_label(v):
        a, b = v['_filter_pair'].split('-')
        a, b = int(a), int(b)
        p = v['_filter_passive_listener']

        a_rssi_mean, a_rssi_std = get_rssi_mean_std(p, a)
        b_rssi_mean, b_rssi_std = get_rssi_mean_std(p, b)

        return "{:.2f} [{:.2f}], {:.2f} [{:.2f}]".format(a_rssi_mean, a_rssi_std, b_rssi_mean, b_rssi_std)

    for k, v in all_df.iterrows():
        ax.annotate("{} {} {}".format(v['_filter_pair'], v['_filter_passive_listener'], get_label(v)),
                    (v['twr_tof_ds_err_std'], v['tdoa_est_ds_err_std']), xytext=(5, -5), textcoords='offset points',
                    family='sans-serif', fontsize=6, color='black')

    plt.axline((0, 0), slope=np.sqrt(2.5 / 0.5))

    plt.show()




def export_twr_vs_tdoa_scatter(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=0)

    all_df = all_df[all_df['tdoa_count'].notna()]

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    #gb = all_df.groupby(by='_filter_pair').agg('median') #'.quantile([0, 0.25, 0.5, 0.75, 0.95, 1])
    # all_df = all_df[['ratio']]
    # all_df['ratio'].boxplot()

    fig, ax = plt.subplots()
    all_df.plot.scatter('twr_tof_ds_err_std', 'tdoa_est_ds_err_std', ax=ax)


    def get_tdoa_dist_a(v):
        a, b = v['_filter_pair'].split('-')
        a, b = int(a), int(b)
        p = v['_filter_passive_listener']

        dist_a = get_dist(trento_b.dev_positions[trento_b.devs[a]], trento_b.dev_positions[trento_b.devs[p]])
        dist_b = get_dist(trento_b.dev_positions[trento_b.devs[b]], trento_b.dev_positions[trento_b.devs[p]])

        return dist_a

    def get_tdoa_dist_b(v):
        a, b = v['_filter_pair'].split('-')
        a, b = int(a), int(b)
        p = v['_filter_passive_listener']

        dist_a = get_dist(trento_b.dev_positions[trento_b.devs[a]], trento_b.dev_positions[trento_b.devs[p]])
        dist_b = get_dist(trento_b.dev_positions[trento_b.devs[b]], trento_b.dev_positions[trento_b.devs[p]])

        return dist_b

    for k, v in all_df.iterrows():
        ax.annotate("{} {} ({:.2f}, {:.2f}, {:.2f})".format(v['_filter_pair'], v['_filter_passive_listener'], get_tdoa_dist_a(v), get_tdoa_dist_b(v), v['tdoa']), (v['twr_tof_ds_err_std'], v['tdoa_est_ds_err_std']),  xytext=(5, -5), textcoords='offset points', family='sans-serif', fontsize=6, color='black')

    plt.axline((0,0), slope=np.sqrt(2.5/0.5))

    plt.show()

    print(all_df)
    exit()


def export_new_twr_variance_based_model_for_tdoa(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)

    var_df = twr_df.groupby('pair').agg({'twr_tof_ss': 'var'})
    var_dict = var_df.to_dict()['twr_tof_ss']

    def compute_exp_std(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        return np.sqrt(var_dict["{}-{}".format(b, a)] + 2*var_dict["{}-{}".format(a, p)] + 2*var_dict["{}-{}".format(b, p)])

    rx_noise_df = get_cached_rx_noise(trento_b, 'exp_rx_noise_10039', bias_corrected=True, skip_to_round=50,
                                 up_to_round=None)

    rx_var_dict = {}
    for tx in range(len(trento_b.devs)):
        for rx in range(len(trento_b.devs)):
            if tx != rx:
                m_rx_std = rx_noise_df[(rx_noise_df['tx_number'] == tx) & (rx_noise_df['rx_number'] == rx)]['rx_std_est'].median()
                rx_var_dict[(tx, rx)] = (m_rx_std * m_rx_std)


    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=0)
    all_df = all_df[all_df['tdoa_count'].notna()]

    # CLEANUP
    #all_df = all_df[all_df['_filter_passive_listener'] != 0]
    #for b in range(len(trento_b.devs)):
    #    all_df = all_df[all_df['_filter_pair'] != "{}-{}".format(0, b)]
    #    all_df = all_df[all_df['_filter_pair'] != "{}-{}".format(b, 0)]

    def construct_coeff(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        return [rx_var_dict[(a, b)], rx_var_dict[(b, a)], rx_var_dict[(a, p)], rx_var_dict[(b, p)], 1]


    k = 'tdoa_est_ss_both_err_std'
    coeff = np.asarray([construct_coeff(x['_filter_pair'], x['_filter_passive_listener']) for (i, x) in all_df.iterrows()])
    ordinate = np.asarray([r[k]*r[k] for (i, r) in all_df.iterrows()])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    #ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((all_df[k]*all_df[k]-(all_df[k]*all_df[k]).mean()) * (all_df[k]*all_df[k]-(all_df[k]*all_df[k]).mean())).sum()

    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()


    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()

    all_df['expected_std'] = all_df.apply(lambda x: compute_exp_std(x['_filter_pair'], x['_filter_passive_listener']), axis=1)
    all_df.plot.scatter('tdoa_est_ds_err_std', 'expected_std', ax=ax)

    for k, v in all_df.iterrows():
        ax.annotate("{}".format(v['_filter_pair']), (v['tdoa_est_ds_err_std'], v['expected_std']),  xytext=(5, -5), textcoords='offset points', family='sans-serif', fontsize=6, color='black')

    plt.axline((0,0), slope=1.0)

    all_df['ss_res'] = (all_df['tdoa_est_ds_err_std']-all_df['expected_std']) * (all_df['tdoa_est_ds_err_std']-all_df['expected_std'])
    all_df['ss_tot'] = (all_df['tdoa_est_ds_err_std']-all_df['tdoa_est_ds_err_std'].mean()) * (all_df['tdoa_est_ds_err_std']-all_df['tdoa_est_ds_err_std'].mean())

    r2 = 1- all_df['ss_res'].sum() / all_df['ss_tot'].sum()
    print("R2 score", r2)


    rmse = ((all_df['expected_std'] - all_df['tdoa_est_ds_err_std']) ** 2).mean() ** .5
    print("rmse", rmse)


    plt.show()

    print(all_df)
    exit()


def export_new_twr_variance_based_model_for_tof(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    skip_to_round=50

    rx_noise_df = get_cached_rx_noise(trento_b, 'exp_rx_noise_10039', bias_corrected=True, skip_to_round=skip_to_round)

    twr_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round)

    rx_var_dict = {}
    for tx in range(len(trento_b.devs)):
        for rx in range(len(trento_b.devs)):
            if tx != rx:
                m_rx_std = rx_noise_df[(rx_noise_df['tx_number'] == tx) & (rx_noise_df['rx_number'] == rx)]['rx_std_est'].median()
                rx_var_dict[(tx, rx)] = (m_rx_std * m_rx_std)

    # CLEANUP
    #all_df = all_df[all_df['_filter_passive_listener'] != 0]
    #for b in range(len(trento_b.devs)):
    #    all_df = all_df[all_df['_filter_pair'] != "{}-{}".format(0, b)]
    #    all_df = all_df[all_df['_filter_pair'] != "{}-{}".format(b, 0)]

    def construct_coeff(pair):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        return [rx_var_dict[(a, b)]*0.125, rx_var_dict[(b, a)]*0.25]


    k = 'twr_tof_ds_err_std'
    coeff = np.asarray([construct_coeff(x['_filter_pair']) for (i, x) in twr_df.iterrows()])
    ordinate = np.asarray([r[k]*r[k] for (i, r) in twr_df.iterrows()])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    #ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean()) * (twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean())).sum()

    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()



def export_final_twr_variance_based_model(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    skip_to_round=50

    twr_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round)

    # we first derive rx noise from the twr results

    def construct_sym_coeff(pair, passive=None):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        base_rx_noise = np.zeros(shape=(round((len(trento_b.devs) * (len(trento_b.devs)-1))/2)))
        base_rx_noise[pair_index(b, a)] = 0.25+0.125
        return base_rx_noise

    def construct_asym_coeff(pair, passive=None):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        base_rx_noise = np.zeros(shape=(len(trento_b.devs)* (len(trento_b.devs)-1)))
        def calc_index(tx, rx):
            base = rx*(len(trento_b.devs)-1)

            assert tx != rx
            if tx > rx:
                tx -= 1

            return base+tx
        assert calc_index(b, a) != calc_index(a, b)

        base_rx_noise[calc_index(b, a)] = 0.25
        base_rx_noise[calc_index(a, b)] = 0.125
        return base_rx_noise


    twr_df = twr_df[twr_df['_filter_passive_listener'] == 0]

    k = 'twr_tof_ds_err_std'
    coeff = np.asarray([construct_sym_coeff(x['_filter_pair']) for (i, x) in twr_df.iterrows()])
    ordinate = np.asarray([r[k]*r[k] for (i, r) in twr_df.iterrows()])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    print(sum_of_squared_residuals)
    #ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean()) * (twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean())).sum()
    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()

def export_final_tdoa_variance_based_model(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    skip_to_round=50

    twr_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round)

    # we first derive rx noise from the twr results

    def construct_sym_coeff(pair, passive=None):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        base_rx_noise = np.zeros(shape=(round((len(trento_b.devs) * (len(trento_b.devs)-1))/2)))
        base_rx_noise[pair_index(b, a)] = 0.25+0.125
        return base_rx_noise

    def construct_asym_coeff(pair, passive=None):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        base_rx_noise = np.zeros(shape=(len(trento_b.devs)* (len(trento_b.devs)-1)))
        def calc_index(tx, rx):
            base = rx*(len(trento_b.devs)-1)

            assert tx != rx
            if tx > rx:
                tx -= 1

            return base+tx
        assert calc_index(b, a) != calc_index(a, b)

        base_rx_noise[calc_index(b, a)] = 0.25
        base_rx_noise[calc_index(a, b)] = 0.125
        base_rx_noise[calc_index(a, p)] = 1
        base_rx_noise[calc_index(b, p)] = 1
        return base_rx_noise

    k = 'twr_tof_ds_err_std'
    coeff = np.asarray([construct_sym_coeff(x['_filter_pair'], x['_filter_passive_listener']) for (i, x) in twr_df.iterrows()])
    ordinate = np.asarray([r[k]*r[k] for (i, r) in twr_df.iterrows()])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    print(sum_of_squared_residuals)
    #ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean()) * (twr_df[k]*twr_df[k]-(twr_df[k]*twr_df[k]).mean())).sum()
    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()

def export_new_twr_variance_based_model_using_ss_diff(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)

    skip_to_round = 50
    twr_df = twr_df[twr_df['round'] >= skip_to_round]

    twr_df['twr_ss_diff'] = twr_df['twr_tof_ss']-twr_df['twr_tof_ss_reverse']
    twr_df['twr_ss_diff_ds'] = twr_df['twr_tof_ss']-twr_df['twr_tof_ds']
    twr_df['twr_ss_reverse_diff_ds'] = twr_df['twr_tof_ss_reverse']-twr_df['twr_tof_ds']

    var_df = twr_df.groupby('pair').agg({'twr_ss_diff': 'var', 'twr_ss_diff_ds': 'var', 'twr_ss_reverse_diff_ds': 'var', 'twr_tof_ds': 'var', 'twr_tof_ss': 'var'})
    var_dict = var_df.to_dict()

    cfo_noise_corrected_dict = {}

    for pair in var_dict['twr_ss_diff']:
        ss_cfo_noise_var = var_dict['twr_ss_diff_ds'][pair]
        ss_reverse_cfo_noise_var = var_dict['twr_ss_reverse_diff_ds'][pair]

        cfo_noise_corrected_dict[pair] = var_dict['twr_ss_diff'][pair]-ss_cfo_noise_var-ss_reverse_cfo_noise_var
        print(cfo_noise_corrected_dict[pair])
    exit()
    # The variance we get is 0.5 the one we need

    def compute_exp_std(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)




        #return np.sqrt(0.5*var_dict["{}-{}".format(a, b)] + 0.5*var_dict["{}-{}".format(b, a)] + 2*var_dict["{}-{}".format(a, p)] + 2*var_dict["{}-{}".format(b, p)])
        return np.sqrt(0.5*var_dict["{}-{}".format(a, b)] + 0.5*var_dict["{}-{}".format(b, a)])

    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round)
    all_df = all_df[all_df['tdoa_count'].notna()]

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()

    all_df['expected_std'] = all_df.apply(lambda x: compute_exp_std(x['_filter_pair'], x['_filter_passive_listener']), axis=1)

    compare_axis_y = 'twr_tof_ss_err_std'
    compare_axis_x = 'expected_std'

    all_df.plot.scatter(compare_axis_y, compare_axis_x, ax=ax)

    for k, v in all_df.iterrows():
        ax.annotate("{}".format(v['_filter_pair']), (v[compare_axis_y], v['expected_std']),  xytext=(5, -5), textcoords='offset points', family='sans-serif', fontsize=6, color='black')

    plt.axline((0,0), slope=1.0)

    all_df['ss_res'] = (all_df[compare_axis_y]-all_df['expected_std']) * (all_df[compare_axis_y]-all_df['expected_std'])

    print("mean", (all_df[compare_axis_y]-all_df['expected_std']).mean())

    all_df['ss_tot'] = (all_df[compare_axis_y]-all_df[compare_axis_y].mean()) * (all_df[compare_axis_y]-all_df[compare_axis_y].mean())

    r2 = 1- all_df['ss_res'].sum() / all_df['ss_tot'].sum()
    print("R2 score", r2)

    plt.show()

    print(all_df)


def export_new_twr_variance_based_model_using_ds_diff(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)

    twr_df['twr_ds_diff'] = twr_df['round_a'] - twr_df['relative_drift']*twr_df['delay_b'] - twr_df['relative_drift']*twr_df['round_b'] + twr_df['delay_a']


    var_df = twr_df.groupby('pair').agg({'twr_ds_diff': 'var'})
    var_dict = var_df.to_dict()['twr_ds_diff']

    # The variance we get is 0.5 the one we need

    print(var_dict)
    exit()
    def compute_exp_std(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        return np.sqrt(0.5*var_dict["{}-{}".format(a, b)] + 0.5*var_dict["{}-{}".format(b, a)] + 2*var_dict["{}-{}".format(a, p)] + 2*var_dict["{}-{}".format(b, p)])

    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=0)
    all_df = all_df[all_df['tdoa_count'].notna()]

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()

    all_df['expected_std'] = all_df.apply(lambda x: compute_exp_std(x['_filter_pair'], x['_filter_passive_listener']), axis=1)
    all_df.plot.scatter('tdoa_est_ss_both_err_std', 'expected_std', ax=ax)

    for k, v in all_df.iterrows():
        ax.annotate("{}".format(v['_filter_pair']), (v['tdoa_est_ss_both_err_std'], v['expected_std']),  xytext=(5, -5), textcoords='offset points', family='sans-serif', fontsize=6, color='black')

    plt.axline((0,0), slope=1.0)

    all_df['ss_res'] = (all_df['tdoa_est_ss_both_err_std']-all_df['expected_std']) * (all_df['tdoa_est_ss_both_err_std']-all_df['expected_std'])
    all_df['ss_tot'] = (all_df['tdoa_est_ss_both_err_std']-all_df['tdoa_est_ss_both_err_std'].mean()) * (all_df['tdoa_est_ss_both_err_std']-all_df['tdoa_est_ss_both_err_std'].mean())

    r2 = 1- all_df['ss_res'].sum() / all_df['ss_tot'].sum()
    print("R2 score", r2)

    plt.show()

    print(all_df)

def export_new_twr_variance_based_model_with_cfo_extraction(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    twr_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=use_bias_correction)



    var_df = twr_df.groupby('pair').agg({'twr_tof_ds': 'var'})
    var_dict = var_df.to_dict()['twr_tof_ds']

    def compute_exp_std(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)
        return np.sqrt(var_dict["{}-{}".format(b, a)] + 2 * var_dict["{}-{}".format(a, p)] + 2 * var_dict[
            "{}-{}".format(b, p)])

    all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=0)
    all_df = all_df[all_df['tdoa_count'].notna()]

    from matplotlib import cm
    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()

    all_df['expected_std'] = all_df.apply(
        lambda x: compute_exp_std(x['_filter_pair'], x['_filter_passive_listener']), axis=1)
    all_df.plot.scatter('tdoa_est_ds_err_std', 'expected_std', ax=ax)

    for k, v in all_df.iterrows():
        ax.annotate("{}".format(v['_filter_pair']), (v['tdoa_est_ds_err_std'], v['expected_std']), xytext=(5, -5),
                    textcoords='offset points', family='sans-serif', fontsize=6, color='black')

    plt.axline((0, 0), slope=1.0)

    all_df['ss_res'] = (all_df['tdoa_est_ds_err_std'] - all_df['expected_std']) * (
                all_df['tdoa_est_ds_err_std'] - all_df['expected_std'])
    all_df['ss_tot'] = (all_df['tdoa_est_ds_err_std'] - all_df['tdoa_est_ds_err_std'].mean()) * (
                all_df['tdoa_est_ds_err_std'] - all_df['tdoa_est_ds_err_std'].mean())

    r2 = 1 - all_df['ss_res'].sum() / all_df['ss_tot'].sum()
    print("R2 score", r2)

    rmse = ((all_df['expected_std'] - all_df['tdoa_est_ds_err_std']) ** 2).mean() ** .5
    print("rmse", rmse)

    plt.show()

    print(all_df)
    exit()

def export_twr_scatter_dist(config, export_dir):
    log = 'job_tdma_very_long'

    all_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=True)
    all_df['twr_tof_ds_err'] = all_df['twr_tof_ds'] - all_df['dist']
    all_df = all_df.groupby('pair').agg({'dist': 'max', 'pair': 'max', 'twr_tof_ds_err': 'std'})

    fig, ax = plt.subplots()
    all_df.plot.scatter('dist', 'twr_tof_ds_err', ax=ax)

    for k, v in all_df.iterrows():
        ax.annotate(v['pair'], (v['twr_tof_ds_err'], v['dist']),  xytext=(5, -5), textcoords='offset points', family='sans-serif', fontsize=6, color='black')
    plt.show()




def export_testbed_ds_vs_cfo_comparison(config, export_dir):

    # We can skip several rounds here
    skip_to_round = 0
    up_to_round = None
    log = 'job_tdma_long' # TOOD: export with new data!!!

    filter_above_std = None

    use_bias_correction = False
    vals = compute_means_and_stds(log, None, None, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round, up_to_round=up_to_round)

    ys = [
        vals['twr_tof_ds_mae']*100.0,
        vals['twr_tof_ss_mae']*100.0,
        vals['tdoa_est_ds_mae']*100.0,
        vals['tdoa_est_mixed_mae']*100.0,
        vals['tdoa_est_ss_both_mae']*100.0,
        vals['tdoa_est_ss_init_mae']*100.0,
        vals['tdoa_est_ss_final_mae']*100.0,
    ]
    
    stds = [
        vals['twr_tof_ds_err_std']*100.0,
        vals['twr_tof_ss_err_std']*100.0,
        vals['tdoa_est_ds_err_std']*100.0,
        vals['tdoa_est_mixed_err_std']*100.0,
        vals['tdoa_est_ss_both_err_std']*100.0,
        vals['tdoa_est_ss_init_err_std']*100.0,
        vals['tdoa_est_ss_final_err_std']*100.0,
    ]

    labels = [
        "ToF\nDS",
        "ToF\nSS",
        "TDoA\nDS",
        "TDoA\nMixed",
        "TDoA\nSS-Both",
        "TDoA\nSS-Init",
        "TDoA\nSS-Final",
    ]

    colors=[
        'C0',
        'C0',
        'C1',
        'C1',
        'C1',
        'C1',
        'C1'
    ]

    plt.clf()

    ax = plt.gca()
    plt.bar(labels, ys, yerr=stds, width=0.8, color=colors)

    plt.ylim(0.0, 25.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Measurement Type")
    ax.set_ylabel("Mean Absolute Error [cm]")

    stds_as_np = np.array(stds)
    stds_as_np = np.ravel(stds_as_np, order='C')

    counter = 0
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            height = 0

        ax.text(p.get_x() + p.get_width() / 2., height + stds_as_np[counter],
                "{:.1f}\n[{:.1f}]".format(height, stds_as_np[counter]), fontsize=9, color='black', ha='center',
                va='bottom')

        counter += 1

    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/mae_comparison.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)
    plt.close()


def export_rssi_bias_correction_comparison(config, export_dir):
    # We can skip several rounds here
    skip_to_round = 0
    up_to_round = None
    log = 'job_tdma_long'  # TOOD: export with new data!!!

    use_bias_correction = False
    corrected_vals = compute_means_and_stds(log, None, None, use_bias_correction=True, skip_to_round=skip_to_round, up_to_round=up_to_round)
    uncorrected_vals = compute_means_and_stds(log, None, None, use_bias_correction=False, skip_to_round=skip_to_round, up_to_round=up_to_round)

    labels = [
        "ToF\nDS",
        "ToF\nSS",
        "TDoA\nDS",
        "TDoA\nMixed",
        #"TDoA\nSS-Both",
        #"TDoA\nSS-Init",
        #"TDoA\nSS-Final",
    ]

    ys = {
        'Uncorrected': [
                uncorrected_vals['twr_tof_ds_mae'] * 100.0,
                uncorrected_vals['twr_tof_ss_mae'] * 100.0,
                uncorrected_vals['tdoa_est_ds_mae'] * 100.0,
                uncorrected_vals['tdoa_est_mixed_mae'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_both_mae'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_init_mae'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_final_mae'] * 100.0,
        ],
        'Corrected':
            [
                corrected_vals['twr_tof_ds_mae'] * 100.0,
                corrected_vals['twr_tof_ss_mae'] * 100.0,
                corrected_vals['tdoa_est_ds_mae'] * 100.0,
                corrected_vals['tdoa_est_mixed_mae'] * 100.0,
                # corrected_vals['tdoa_est_ss_both_mae'] * 100.0,
                # corrected_vals['tdoa_est_ss_init_mae'] * 100.0,
                # corrected_vals['tdoa_est_ss_final_mae'] * 100.0,
            ],
    }

    stds = {
        'Corrected':
            [
                corrected_vals['twr_tof_ds_err_std'] * 100.0,
                corrected_vals['twr_tof_ss_err_std'] * 100.0,
                corrected_vals['tdoa_est_ds_err_std'] * 100.0,
                corrected_vals['tdoa_est_mixed_err_std'] * 100.0,
                #corrected_vals['tdoa_est_ss_both_err_std'] * 100.0,
                #corrected_vals['tdoa_est_ss_init_err_std'] * 100.0,
                #corrected_vals['tdoa_est_ss_final_err_std'] * 100.0,
            ],
        'Uncorrected': [
                uncorrected_vals['twr_tof_ds_err_std'] * 100.0,
                uncorrected_vals['twr_tof_ss_err_std'] * 100.0,
                uncorrected_vals['tdoa_est_ds_err_std'] * 100.0,
                uncorrected_vals['tdoa_est_mixed_err_std'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_both_err_std'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_init_err_std'] * 100.0,
                #uncorrected_vals['tdoa_est_ss_final_err_std'] * 100.0,
        ]
    }


    colors = [
        'C0',
        'C0',
        'C1',
        'C1',
        'C1',
        'C1',
        'C1'
    ]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0.5

    plt.clf()

    ax = plt.gca()

    for attribute, measurement in ys.items():
        offset = width * multiplier

        stds_as_np = np.array(stds[attribute])
        print(measurement)
        rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=stds_as_np)

        stds_as_np = np.ravel(stds_as_np, order='C')

        counter = 0
        for p in rects:
            height = p.get_height()
            if np.isnan(height):
                height = 0

            ax.text(p.get_x() + p.get_width() / 2., height + stds_as_np[counter],
                    "{:.1f}\n[{:.1f}]".format(height, stds_as_np[counter]), fontsize=9, color='black', ha='center',
                    va='bottom')

            counter += 1
        multiplier += 1

    plt.ylim(0.0, 30.0)
    ax.set_axisbelow(True)
    ax.set_xticks(x + width, labels )
    ax.set_xlabel("Measurement Type")
    ax.set_ylabel("Mean Absolute Error [cm]")
    ax.legend()




    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.0, 5.5)
    plt.tight_layout()

    plt.savefig("{}/rssi_bias_correction_comparison.pdf".format(export_dir), bbox_inches='tight', pad_inches=0)
    plt.close()


def export_localization_performance(config, export_dir):
    import sim_experimental_localization

    skip_to_round = 0
    up_to_round = None  # we will use all available measurements for now.
    log = 'job_tdma_long'  # TOOD: export with new data!!!
    use_bias_correction = True

    true_positions = {}
    for k in trento_b.dev_positions:
        true_positions[k] = np.asarray([trento_b.dev_positions[k][0], trento_b.dev_positions[k][1]])

    active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
                                                               use_bias_correction, skip_to_round=skip_to_round,
                                                               up_to_round=up_to_round)
    def proc():
        tof_meas = {}
        tdoa_measurements = {}

        for na, dev_a in enumerate(trento_b.devs):
            for nb, dev_b in enumerate(trento_b.devs):
                if dev_a != dev_b:
                    filtered_active_df = active_df[(active_df['pair'] == "{}-{}".format(na, nb))]

                    tof_meas["{}-{}".format(dev_a, dev_b)] = filtered_active_df['twr_tof_ds'].mean()

                    for nc, dev_c in enumerate(trento_b.devs):
                        if dev_a != dev_c and dev_c != dev_b:
                            # add passive measurement
                            filtered_passive_df = passive_df[(passive_df['pair'] == "{}-{}".format(na, nb)) & (passive_df['tdoa_device'] == nc)]
                            tdoa_measurements["{}-{}-{}".format(dev_a, dev_b, dev_c)] = filtered_passive_df['tdoa_est_ds'].mean()
        return [tof_meas, tdoa_measurements]
    string_keyed_tof_meas, string_keyed_tdoa_measurements = cached(('export_localization_performance', log, use_bias_correction, skip_to_round, up_to_round, 3), proc)

    tof_meas = {}
    tdoa_measurements = {}
    for k in string_keyed_tof_meas:
        tof_meas[tuple(k.split("-"))] = string_keyed_tof_meas[k]

    for k in string_keyed_tdoa_measurements:
        tdoa_measurements[tuple(k.split("-"))] = string_keyed_tdoa_measurements[k]

    import itertools


    all_devices = trento_b.devs
    def gen_random_scenarios(num_anchors, num_active):

        # choose different anchors
        anchor_subsets = itertools.combinations(all_devices, r=num_anchors)
        for anchors in anchor_subsets:
            active_subsets = itertools.combinations([d for d in all_devices if d not in anchors], r=num_active)

            for actives in active_subsets:
                passives = [d for d in all_devices if d not in anchors and d not in actives]
                yield anchors, actives, passives

    init_noise_std = 5.0


    def exec_sims(num_anchors, num_active):
        g = gen_random_scenarios(num_anchors, num_active)

        for (anchor_devices, active_devices, passive_devices) in g:

            # (anchor_devices, active_devices, passive_devices) = (('dwm1001.160', 'dwm1001.161', 'dwm1001.168'), (
            # 'dwm1001.162', 'dwm1001.163', 'dwm1001.164', 'dwm1001.165', 'dwm1001.166', 'dwm1001.167', 'dwm1001.169',
            # 'dwm1001.170', 'dwm1001.171', 'dwm1001.172', 'dwm1001.173'), [])


            res = {}

            active_errs, passive_errs = sim_experimental_localization.least_squares_loc(
                true_positions, anchor_devices, active_devices, passive_devices, tof_meas, tdoa_measurements,
                use_cooperative=False, use_tdoa_for_active=False, init_noise_std=init_noise_std)

            res['uncooperative_no_passive_active_rmse'] = np.sqrt(np.mean(active_errs ** 2))
            res['uncooperative_no_passive_passive_rmse'] = np.sqrt(np.mean(passive_errs ** 2))
            res['uncooperative_no_passive_all_rmse'] = np.sqrt(np.mean(np.concatenate((active_errs, passive_errs)) ** 2))

            active_errs, passive_errs = sim_experimental_localization.least_squares_loc(
                true_positions, anchor_devices, active_devices, passive_devices, tof_meas, tdoa_measurements,
                use_cooperative=False, use_tdoa_for_active=True, init_noise_std=init_noise_std)

            res['uncooperative_passive_active_rmse'] = np.sqrt(np.mean(active_errs ** 2))
            res['uncooperative_passive_passive_rmse'] = np.sqrt(np.mean(passive_errs ** 2))
            res['uncooperative_passive_all_rmse'] =np.sqrt(np.mean(np.concatenate((active_errs, passive_errs)) ** 2))

            active_errs, passive_errs = sim_experimental_localization.least_squares_loc(
                true_positions,anchor_devices, active_devices, passive_devices, tof_meas, tdoa_measurements,
                use_cooperative=True, use_tdoa_for_active=False, init_noise_std=init_noise_std)

            res['cooperative_no_passive_active_rmse'] = np.sqrt(np.mean(active_errs ** 2))
            res['cooperative_no_passive_passive_rmse'] = np.sqrt(np.mean(passive_errs ** 2))
            res['cooperative_no_passive_all_rmse'] = np.sqrt(np.mean(np.concatenate((active_errs, passive_errs)) ** 2))

            active_errs, passive_errs = sim_experimental_localization.least_squares_loc(
                true_positions, anchor_devices, active_devices, passive_devices, tof_meas, tdoa_measurements,
                use_cooperative=True, use_tdoa_for_active=True, init_noise_std=init_noise_std)

            res['cooperative_no_passive_active_rmse'] = np.sqrt(np.mean(active_errs ** 2))
            res['cooperative_no_passive_passive_rmse'] = np.sqrt(np.mean(passive_errs ** 2))
            res['cooperative_no_passive_all_rmse'] = np.sqrt(np.mean(np.concatenate((active_errs, passive_errs)) ** 2))


            yield res

    def sim_all(limit=None):
        anchor_nums = [3]
        for x1 in anchor_nums:
            for x2 in reversed(range(len(all_devices)-x1+1)):
                for res in itertools.islice(exec_sims(x1, x2), 0, limit):
                    res['num_anchors'] = x1
                    res['num_active'] = x2
                    res['num_passive'] = len(all_devices)-(x1+x2)
                    yield res

    for r in sim_all():
        print(r)

    pass


def export_histogram_std(config, export_dir):
    # We can skip several rounds here
    skip_to_round = 0
    up_to_round = None
    log = 'job_tdma_long'  # TOOD: export with new data!!!

    filter_above_std = None

    for use_bias_correction in [True, False]:
        vals = compute_means_and_stds(log, None, None, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round,
                                      up_to_round=up_to_round)

        active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
                                                                   use_bias_correction, skip_to_round=skip_to_round,
                                                                   up_to_round=up_to_round)

        plt.clf()
        ax = plt.gca()

        active_std_df = active_df.groupby('pair').agg('std')
        active_std_df['twr_tof_ds'] *= 100.0
        #active_std_df['twr_tof_ds_err'] *= 100.0
        #active_std_df.hist(column='twr_tof_ds', bins=100, ax=ax)
        active_std_df.hist(column='twr_tof_ds', bins=100, ax=ax)

        ax.set_axisbelow(True)
        ax.set_xlabel("DS-TWR SD [cm]")
        ax.set_ylabel("Number of Pairs")
        #ax.legend()

        plt.title('')

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/histogram_ds_twr_bias_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.clf()
        ax = plt.gca()

        passive_std_df = passive_df.groupby(['pair', 'tdoa_device']).agg('std')
        passive_std_df['tdoa_est_ds'] *= 100.0
        #passive_std_df['tdoa_est_ds_err'] *= 100.0
        #passive_std_df.hist(column='tdoa_est_ds', bins=100, ax=ax)
        passive_std_df.hist(column='tdoa_est_ds', bins=100, ax=ax)
        plt.title('')

        ax.set_axisbelow(True)
        ax.set_xlabel("DS-TDoA SD [cm]")
        ax.set_ylabel("Number of Triples")
        #ax.legend()

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/histogram_ds_tdoa_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.clf()
        ax = plt.gca()

        active_std_df.hist(column='twr_tof_ds', bins=50, ax=ax, label='DS TWR', alpha=0.75)
        passive_std_df.hist(column='tdoa_est_ds', bins=100, ax=ax, label='DS TDoA', alpha=0.75)

        ax.set_axisbelow(True)
        ax.set_xlabel("Sample SD [cm]")
        ax.set_ylabel("Number of Pairs / Triples")
        # ax.legend()

        plt.title('')

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/std_histogram_ds_active_passive_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()


def export_histogram_mean(config, export_dir):
    # We can skip several rounds here
    skip_to_round = 0
    up_to_round = None
    log = 'job_tdma_long'  # TOOD: export with new data!!!

    filter_above_std = None

    for use_bias_correction in [True, False]:
        vals = compute_means_and_stds(log, None, None, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round,
                                      up_to_round=up_to_round)

        active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
                                                                   use_bias_correction, skip_to_round=skip_to_round,
                                                                   up_to_round=up_to_round)

        plt.clf()
        ax = plt.gca()

        active_std_df = active_df.groupby('pair').agg('mean')
        active_std_df['twr_tof_ds_err'] *= 100.0
        #active_std_df['twr_tof_ds_err'] *= 100.0
        #active_std_df.hist(column='twr_tof_ds', bins=100, ax=ax)
        active_std_df.hist(column='twr_tof_ds_err', bins=100, ax=ax)

        ax.set_axisbelow(True)
        ax.set_xlabel("DS-TWR SD [cm]")
        ax.set_ylabel("Number of Pairs")
        #ax.legend()

        plt.title('')

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/histogram_ds_twr_bias_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.clf()
        ax = plt.gca()

        passive_std_df = passive_df.groupby(['pair', 'tdoa_device']).agg('mean')
        passive_std_df['tdoa_est_ds_err'] *= 100.0
        #passive_std_df['tdoa_est_ds_err'] *= 100.0
        #passive_std_df.hist(column='tdoa_est_ds', bins=100, ax=ax)
        passive_std_df.hist(column='tdoa_est_ds_err', bins=100, ax=ax)
        plt.title('')

        ax.set_axisbelow(True)
        ax.set_xlabel("DS-TDoA SD [cm]")
        ax.set_ylabel("Number of Triples")
        #ax.legend()

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/histogram_ds_tdoa_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()
        plt.clf()
        ax = plt.gca()

        active_std_df.hist(column='twr_tof_ds_err', bins=50, ax=ax, label='DS TWR', alpha=0.75)
        passive_std_df.hist(column='tdoa_est_ds_err', bins=100, ax=ax, label='DS TDoA', alpha=0.75)

        ax.set_axisbelow(True)
        ax.set_xlabel("Sample SD [cm]")
        ax.set_ylabel("Number of Pairs / Triples")
        # ax.legend()

        plt.title('')

        plt.gcf().set_size_inches(6.0, 5.5)
        plt.tight_layout()

        plt.savefig("{}/mean_histogram_ds_active_passive_{}.pdf".format(export_dir, use_bias_correction), bbox_inches='tight', pad_inches=0)
        plt.close()


def get_cached_rx_noise(testbed, run, bias_corrected=True, skip_to_round = 0, up_to_round = None):
    def proc():
        return logs.estimate_rx_noise_using_cfo(testbed, run, bias_corrected=bias_corrected, skip_to_round=skip_to_round, up_to_round=up_to_round)
    return utility.cached_dt(('get_cached_rx_noise', testbed.name, run, bias_corrected, skip_to_round, up_to_round), proc)


def export_measured_mean_std_matrix(config, export_dir):

    skip_to_round = 50  # 200?
    use_bias_correction = True
    log = 'job_tdma_long'

    for metric in ['twr_tof_ds_err_std', 'twr_tof_ds_err_mean']: #, 'twr_tof_ss_err_std', 'twr_tof_ss_err_mean']:

        for (c, t) in enumerate([trento_b]):

            all_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction, skip_to_round=skip_to_round)

            ma = np.zeros((len(t.devs), len(t.devs)))

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if b != a:

                        xs = all_df[(all_df['_filter_pair'] == "{}-{}".format(a,b))][metric]

                        std = xs.median()

                        if std is not None:
                            s = std * 100
                        else:
                            s = np.nan
                        ma[a, b] = s

            print(t.name, "mean std", ma.mean(), "median",np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

            plt.clf()
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.matshow(ma)

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
            fig.set_size_inches(6.0, 6.0)
            plt.tight_layout()

            ax.set_xlabel('RX Device')
            ax.set_ylabel('TX Device')

            plt.savefig("{}/{}_{}.pdf".format(export_dir, metric, t.name), bbox_inches='tight', pad_inches=0)

            plt.close()

def export_measured_rx_noise(config, export_dir):

    skip_to_round = 50
    up_to_round = None
    use_bias_correction = True
    log = 'exp_rx_noise_10041'

    # we also directly search for the triples with the lowest rx variance variation

    lowest_var = 100000.0
    lowest_tripel = []

    for (c, t) in enumerate([trento_b]):

        est_df = get_cached_rx_noise(t, log, bias_corrected=use_bias_correction, skip_to_round=skip_to_round, up_to_round=up_to_round)

        ma = np.zeros((len(t.devs), len(t.devs)))

        for a in range(len(t.devs)):
            for b in range(len(t.devs)):
                if b != a:

                    xs = est_df[(est_df['tx_number'] == a) & (est_df['rx_number'] == b)]['rx_std_est']

                    std = xs.median()

                    if std is not None:
                        s = std * 100
                    else:
                        s = np.nan
                    ma[a, b] = s

        tripel_stds = []
        for a in range(len(t.devs)):
            for b in range(len(t.devs)):
                for c in range(len(t.devs)):
                    if b != a and c != a and c != b:
                        stds = np.asarray([
                            ma[a, b],
                            ma[b, a],
                            ma[a, c],
                            ma[c, a],
                            ma[b, c],
                            ma[c, b],
                        ])

                        tripel_stds.append({
                            'tripel': "{}-{}-{}".format(a,b,c),
                            'std': stds.std()
                        })
        tripel_df = pd.DataFrame.from_records(tripel_stds).sort_values('std')
        print(tripel_df.head(20))

        print(t.name, "mean std", ma.mean(), "median", np.median(ma), "max", np.max(ma), "90% quantile", np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma)

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
        fig.set_size_inches(6.0, 6.0)
        plt.tight_layout()

        ax.set_xlabel('RX Device')
        ax.set_ylabel('TX Device')

        plt.savefig("{}/rx_noise_variance_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()


def export_predicted_ds_twr(config, export_dir):
    skip_to_round = 0  # 200? TODO
    up_to_round = 120  # 200?
    use_bias_correction = True
    log = 'exp_rx_noise_10046'

    for (c, t) in enumerate([trento_b]):

        est_df = get_cached_rx_noise(t, log, bias_corrected=use_bias_correction, skip_to_round=skip_to_round,
                                     up_to_round=up_to_round)

        ma = np.zeros((len(t.devs), len(t.devs)))

        for a in range(len(t.devs)):
            for b in range(len(t.devs)):
                if b != a:

                    noise_rxb = est_df[(est_df['tx_number'] == a) & (est_df['rx_number'] == b)]['rx_std_est'].median()
                    noise_rxa = est_df[(est_df['tx_number'] == a) & (est_df['rx_number'] == b)]['rx_std_est'].median()

                    std = np.sqrt(0.25*noise_rxb*noise_rxb+0.25*noise_rxa*noise_rxa)

                    if std is not None:
                        s = std * 100
                    else:
                        s = np.nan
                    ma[a, b] = s

        print(t.name, "mean std", ma.mean(), "median", np.median(ma), "max", np.max(ma), "90% quantile",
              np.quantile(ma, 0.9), "95% quantile", np.quantile(ma, 0.95))

        plt.clf()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(ma)

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

        ax.xaxis.set_major_formatter(lambda x, pos: int(x + 1))
        ax.yaxis.set_major_formatter(lambda x, pos: int(x + 1))
        fig.set_size_inches(6.0, 6.0)
        plt.tight_layout()

        ax.set_xlabel('Responder')
        ax.set_ylabel('Initiator')

        plt.savefig("{}/predicted_ds_twr_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

def export_base_rx_noise_level_tof(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    skip_to_round = 50

    rx_noise_df = get_cached_rx_noise(trento_b, 'exp_rx_noise_10039', bias_corrected=True,
                                      skip_to_round=skip_to_round)

    twr_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction,
                                                   skip_to_round=skip_to_round)

    rx_var_dict = {}
    for tx in range(len(trento_b.devs)):
        for rx in range(len(trento_b.devs)):
            if tx != rx:
                m_rx_std = rx_noise_df[(rx_noise_df['tx_number'] == tx) & (rx_noise_df['rx_number'] == rx)][
                    'rx_std_est'].median()
                rx_var_dict[(tx, rx)] = (m_rx_std * m_rx_std)

    def construct_coeff(pair):
        a, b = pair.split("-")
        a, b = int(a), int(b)

        base_rx_noise = np.zeros(shape=len(trento_b.devs))
        path_noises = np.zeros(shape=round(len(trento_b.devs)*(len(trento_b.devs)-1)/2))

        base_rx_noise[a] = 1
        base_rx_noise[b] = 1

        path_noises[pair_index(a,b)] = 1

        return np.concatenate((base_rx_noise, path_noises))

    #k = 'twr_tof_ds_err_std'
    k = 'twr_tof_ds_err_std'
    coeff = np.asarray([construct_coeff(x['_filter_pair']) for (i, x) in twr_df.iterrows()])
    ordinate = np.asarray([r[k] * r[k] for (i, r) in twr_df.iterrows()])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    # ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((twr_df[k] * twr_df[k] - (twr_df[k] * twr_df[k]).mean()) * (
                twr_df[k] * twr_df[k] - (twr_df[k] * twr_df[k]).mean())).sum()

    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()

def export_base_rx_noise_level_tdoa(config, export_dir):
    log = 'job_tdma_long'

    use_bias_correction = True
    skip_to_round = 50

    rx_noise_df = get_cached_rx_noise(trento_b, 'exp_rx_noise_10039', bias_corrected=True,
                                      skip_to_round=skip_to_round)

    twr_df = cached_compute_all_agg_means_and_stds(log, use_bias_correction=use_bias_correction,
                                                   skip_to_round=skip_to_round)

    rx_var_dict = {}
    for tx in range(len(trento_b.devs)):
        for rx in range(len(trento_b.devs)):
            if tx != rx:
                m_rx_std = rx_noise_df[(rx_noise_df['tx_number'] == tx) & (rx_noise_df['rx_number'] == rx)][
                    'rx_std_est'].median()
                rx_var_dict[(tx, rx)] = (m_rx_std * m_rx_std)

    def construct_coeff(pair, p):
        a, b = pair.split("-")
        a, b = int(a), int(b)

        base_rx_noise = np.zeros(shape=len(trento_b.devs))
        path_noises = np.zeros(shape=round(len(trento_b.devs)*(len(trento_b.devs)-1)/2))

        base_rx_noise[a] = 0.25
        base_rx_noise[b] = 0.25
        path_noises[pair_index(a,b)] = 0.5

        base_rx_noise[p] = 2
        path_noises[pair_index(a,p)] = 1
        path_noises[pair_index(b,p)] = 1

        return np.concatenate((base_rx_noise, path_noises, [1]))


    #k = 'twr_tof_ds_err_std'
    k = 'tdoa_est_ss_init_err_std'
    twr_df = twr_df[twr_df['tdoa_count'].notna()]

    coeff = np.asarray([construct_coeff(x['_filter_pair'], x['_filter_passive_listener']) for (i, x) in twr_df.iterrows()])
    ordinate = np.asarray([r[k] * r[k] for (i, r) in twr_df.iterrows()])
    print(coeff[0:4])
    print(ordinate[0:4])

    x, sum_of_squared_residuals, _, _ = np.linalg.lstsq(coeff, ordinate, rcond=-1)

    # ss_tot = ((all_df[k]-all_df[k].mean()) * (all_df[k]-all_df[k].mean())).sum()
    ss_tot = ((twr_df[k] * twr_df[k] - (twr_df[k] * twr_df[k]).mean()) * (
                twr_df[k] * twr_df[k] - (twr_df[k] * twr_df[k]).mean())).sum()

    r2 = 1 - sum_of_squared_residuals / ss_tot.sum()
    print("R2 score", r2)
    print(x)
    exit()

def export_delay_exp(config, export_dir):

    log = '10158'

    # TODO add passive_df!!
    active_df = get_df(log, tdoa_src_dev_number=None, use_bias_correction=True)
    #active_df, passive_df = extract_active_and_all_passive_dfs(log, None, None,
    #                                                           use_bias_correction=True, skip_to_round=0,
    #                                                           up_to_round=None)

    active_df['delay_b_ms'] = active_df['delay_b'].apply(lambda x : np.round(convert_ts_to_sec(x) * 1000))
    active_df['delay_a_ms'] = active_df['delay_a'].apply(lambda x : np.round(convert_ts_to_sec(x) * 1000))

    print(active_df['delay_a_ms'].unique())
    print(active_df['delay_b_ms'].unique())

    #active_df = active_df[active_df['pair'] == "0-3"]
    active_df_aggr = active_df.groupby('delay_b_ms').agg('count')

    #active_df.plot.scatter(x='delay_b_ms', y='twr_tof_ds_err')
    active_df_aggr.plot.line(y='twr_tof_ds_err')

    plt.show()



if __name__ == '__main__':

    config = load_env_config()

    load_plot_defaults()

    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']

    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    steps = [
        #export_testbed_layouts,
        #export_scatter_graph_trento_a,
        #export_trento_a_pairs,
        #export_simulation_performance,
        #export_filtered_mae_reduction,
        #export_testbed_variance,
        #export_overall_mae_reduction,
        #export_overall_rmse_reduction,
        #export_tdoa_simulation_drift_performance,
        #export_tdoa_simulation_rx_noise
        #export_tdoa_simulation_response_std,
        #export_tdoa_simulation_response_std_scatter,
        #export_testbed_variance,
        #export_testbed_variance_calculated_tof,
        #export_testbed_variance_calculated_tof_ci,
        #export_testbed_tdoa_variance,
        #export_testbed_tdoa_calculated_tdoa_variance,
        #export_testbed_tdoa_calculated_tdoa_ci_variance,
        #export_testbed_variance_calculated_tof,
        #export_testbed_variance_calculated_tof_ci_avg,
        #export_tof_simulation_response_std,
        #export_twr_scatter,
        #export_twr_vs_tdoa_scatter_rssi,
        #export_twr_vs_tdoa_scatter,
        #export_loc_sim,
        #export_twr_scatter_dist
        #export_new_twr_variance_based_model,
        #export_new_twr_variance_based_model
        #export_testbed_ds_vs_cfo_comparison,
        #export_rssi_bias_correction_comparison,
        #export_localization_performance,
        #export_new_twr_variance_based_model_using_ss_diff
        #export_histograms,
        #export_predicted_ds_twr,
        #export_measured_mean_std_matrix,
        #export_measured_rx_noise,
        #export_new_twr_variance_based_model_with_cfo_extractions
        #export_histograms,
        #export_histogram_mean,
        #export_new_twr_variance_based_model_for_tof,
        #export_base_rx_noise_level_tdoa
        export_delay_exp
        #export_final_twr_variance_based_model,
        #export_final_tdoa_variance_based_model
        #export_new_twr_variance_based_model_for_tof
    ]

    #for step in progressbar.progressbar(steps, redirect_stdout=True):
    for step in steps:
        name = step.__name__.removeprefix(METHOD_PREFIX)
        print("Handling {}".format(name))
        export_dir = os.path.join(config['EXPORT_DIR']) + '/'
        os.makedirs(export_dir, exist_ok=True)
        step(config, export_dir)
