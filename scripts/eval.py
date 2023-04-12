import os
import progressbar
import numpy as np

from testbed import lille, trento_a, trento_b

from logs import gen_estimations_from_testbed_run, gen_measurements_from_testbed_run, gen_delay_estimates_from_testbed_run
from base import get_dist, pair_index, convert_ts_to_sec, convert_sec_to_ts, convert_ts_to_m, convert_m_to_ts, ci_to_rd


import matplotlib
import matplotlib.pyplot as plt
from utility import slugify, cached, init_cache, load_env_config

import pandas as pd

METHOD_PREFIX = 'export_'

CONFIDENCE_FILL_COLOR = '0.8'
PERCENTILES_FILL_COLOR = '0.5'
COLOR_MAP = 'tab10'

c_in_air = 299702547.236


runs = {
        'trento_a': 'job',  # [(6,3)],
        'trento_b': 'job',
        'lille': 'job'  # [(11,3), (10,3), (7,1), (5,0)],
}

src_devs = {
        'trento_a': 'dwm1001.1',  # [(6,3)],
        'trento_b': 'dwm1001.160',
        'lille': 'dwm1001-1'  # [(11,3), (10,3), (7,1), (5,0)],
}

def load_plot_defaults():
    # Configure as needed
    plt.rc('lines', linewidth=2.0)
    plt.rc('legend', framealpha=1.0, fancybox=True)
    plt.rc('errorbar', capsize=3)
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', size=11)
    plt.rcParams['axes.axisbelow'] = True


def export_simulation_performance(config, export_dir):

    from sim import get_sim_data_rows

    xs = [16, 64, 256, 1024]
    num_repetitions = 100

    def proc():
        return get_sim_data_rows(xs, num_repetitions)

    data_rows = cached(('sim', xs, num_repetitions), proc)

    df = pd.DataFrame(data_rows)

    # df.plot.bar(x='pair',y=['dist', 'est_distance_uncalibrated', 'est_distance_factory', 'est_distance_calibrated'])
    df = df.rename(columns={"gn_mean": "Gauss-Newton", "tdoa_mean": "TDoA", "our_mean": "Proposed"})

    stds = [df['tdoa_std'], df['gn_std'], df['our_std'], df['speedup_err']]

    plt.clf()

    ax = df.plot.bar(x='num_measurements', y=['TDoA', 'Gauss-Newton', 'Proposed'], yerr=stds, width=0.8)
    plt.ylim(0.0, 14.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Number of Measurement Rounds")
    ax.set_ylabel("Mean RMSE [cm]")

    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            height = 0

        ax.text(p.get_x() + p.get_width() / 2., 0.0, '%.1f' % height, fontsize=10, color='black', ha='center',
                va='bottom')
        # ax.text(p.get_x() + p.get_width()/2., 0.5, '%.2f' % stds[offset], fontsize=12, color='black', ha='center', va='bottom')

    plt.grid(color='lightgray', linestyle='dashed')

    plt.gcf().set_size_inches(6.5, 6.5)
    plt.tight_layout()

    plt.savefig("{}/sim_rmse.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
    #plt.show()

    plt.close()



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
                    if a > b:
                        e = res.loc['{}-{}'.format(a, b), ('estimated_m', 'std')]*100   # in cm
                        ma[a, b] = e
                        ma[b, a] = e
            return ma

        ma = np.array(cached(('meas_var', t.name, runs[t.name], src_devs[t.name], 6), proc))
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
        fig.set_size_inches(5.0, 5.0)
        plt.tight_layout()

        plt.savefig("{}/var_ma_{}.pdf".format(export_dir, t.name), bbox_inches='tight', pad_inches=0)

        plt.close()

        #plt.show()

def export_testbed_variance_from_device(config, export_dir):

    std_upper_lim = 10.0

    for (c, t) in enumerate([lille, trento_a, trento_b]):

        def proc():
            est_df = pd.DataFrame.from_records(gen_estimations_from_testbed_run(t, runs[t.name], src_dev=src_devs[t.name]))

            round = est_df['round'].max()
            est_df = est_df[(est_df['round'] == round)]


            ma = np.zeros((len(t.devs), len(t.devs)))

            for a in range(len(t.devs)):
                for b in range(len(t.devs)):
                    if a > b:
                        res = est_df.loc[est_df['pair'] == '{}-{}'.format(a, b), "var_measurement"]

                        e = 0.0
                        if len(res) > 0:
                            var = (res.to_numpy())[0]
                            if var is not None:
                                e = np.sqrt(var)*100

                        ma[a, b] = e
                        ma[b, a] = e
            print(ma)
            return ma

        ma = np.array(cached(('meas_var_device', t.name, runs[t.name], src_devs[t.name], 7), proc))
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

        #plt.show()



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
        'lille': [None, 10.5, None, 22.5],
        'trento_a': [None, 80, None, 7.5],
        'trento_b': [None, 132, None, 7.5]
    }


    for (c,t) in enumerate([trento_a, trento_b, lille]):
        ys = []
        xs = []
        ns = []

        for k in t.devs:
            pos = t.dev_positions[k]
            xs.append(pos[0])
            ys.append(pos[1])
            n = k.replace("dwm1001.", "").replace("dwm1001-", "")
            ns.append(n)

        plt.clf()


        fig, ax = plt.subplots()


        ax.scatter(xs, ys, color='C'+str(c), marker='o')
        ax.set_aspect('equal', adjustable='box')


        for (a,b) in high_variance_connections[t.name]:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)

        for a in always_drawn[t.name]:
            for b in range(len(t.devs)):
                if a > a:
                    plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'r', zorder=-10)


        for i, txt in enumerate(ns):
            ax.annotate(txt, (xs[i], ys[i]), xytext=(3, 0), textcoords='offset points', va='center', ha='left')

        ax.set_axisbelow(True)
        ax.set_xlabel("Position X [m]")
        ax.set_ylabel("Position Y [m]")

        if t.name == 'lille':
            ax.invert_yaxis()

        plt.axis(limits[t.name])

        fig.set_size_inches(4.0, 3.5)
        plt.tight_layout()

        plt.savefig("{}/layout_{}.pdf".format(export_dir, t.name),bbox_inches = 'tight', pad_inches = 0)

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

    scenarios = ["Trento (7)", "Trento (14)", "IoT-Lab (14)"]

    x = np.arange(len(scenarios))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0


    plt.clf()
    fig, ax = plt.subplots(layout='constrained')

    labels = {
        'err_uncalibrated': "Uncalibrated",
        'err_factory': "Factory",
        'err_calibrated': "Calibrated"
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

    scenarios = ["Trento (7)", "Trento (14)", "IoT-Lab (14)"]

    x = np.arange(len(scenarios))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    plt.clf()
    fig, ax = plt.subplots(layout='constrained')

    labels = {
        'err_uncalibrated': "Uncalibrated",
        'err_factory': "Factory",
        'err_calibrated': "Calibrated",
        'err_calibrated_filtered': "Calibrated Filtered",
    }

    for attribute, measurement in errs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=labels[attribute], yerr=stds[attribute])
        ax.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    ax.set_ylabel('MAE [cm]')
    ax.set_xlabel('Scenario')
    ax.set_xticks(x + width, scenarios)
    ax.legend()
    #ax.set_ylim(0, 250)

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

        est_df['Uncalibrated'] = est_df['est_distance_uncalibrated'] - est_df['dist']
        est_df['Factory'] = est_df['est_distance_factory'] - est_df['dist']
        est_df['Calibrated'] = est_df['est_distance_calibrated'] - est_df['dist']
        est_df['err_calibrated_filtered'] = est_df['est_distance_calibrated_filtered'] - est_df['dist']

        est_df = est_df.sort_values(by='Uncalibrated')



        def rename_pairs(x=None):
            return '{}-{}'.format(x['initiator']+1, x['responder']+1)

        est_df['pair'] = est_df.apply(rename_pairs, axis=1)

        # df.plot.bar(x='pair',y=['dist', 'est_distance_uncalibrated', 'est_distance_factory', 'est_distance_calibrated'])
        est_df.plot.bar(x='pair', y=['Uncalibrated', 'Factory', 'Calibrated'])

        plt.legend()
        plt.xlabel("Pair")
        plt.ylabel("Error [cm]")

        plt.gcf().set_size_inches(5.0, 4.0)
        plt.tight_layout()

        plt.gca().yaxis.grid(True, color='lightgray', linestyle='dashed')

        plt.savefig("{}/error_sorted.pdf".format(export_dir), bbox_inches = 'tight', pad_inches = 0)
        plt.close()





if __name__ == '__main__':

    config = load_env_config()

    load_plot_defaults()

    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']

    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    steps = [
        export_simulation_performance,
        export_trento_a_pairs,
        export_testbed_variance,
        #export_testbed_variance_from_device,
        export_overall_mae_reduction,
        export_overall_rmse_reduction,
        #export_testbed_layouts,
    ]

    for step in progressbar.progressbar(steps, redirect_stdout=True):
        name = step.__name__.removeprefix(METHOD_PREFIX)
        print("Handling {}".format(name))
        export_dir = os.path.join(config['EXPORT_DIR']) + '/'
        os.makedirs(export_dir, exist_ok=True)
        step(config, export_dir)
