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

    data_rows = cached_legacy(('export_tdoa_simulation_response_mean', limit, step, 10, num_sims, resp_delay_s), proc)

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

    data_rows = cached_legacy(('export_tdoa_simulation_response_std_scatter', hash(json.dumps(list(xs))), 9, num_sims, resp_delay_s, num_sims), proc)
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


def export_tdoa_simulation_response_std(export_dir):

    from sim_tdoa import sim

    ratio_limit = 0.001
    num = 50

    num_sim_per_rep = 2000
    node_drift_std = 10.0/1000000.0
    mitigate_drift = True
    rx_noise_std = 1.0e-09
    drift_rate_std= 8.0e-08

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
            + (0.5 * a_b_std * (delay_b/comb_delay-1)) ** 2
            + (0.5 * a_b_std * (delay_b/comb_delay)) ** 2
            + (a_p_std * (1-delay_b/comb_delay)) ** 2
            + b_p_std ** 2
            + (a_p_std * (delay_b/comb_delay)) ** 2
        )

    #@utility.cached
    def proc_simulation_response_std(
            ratio_limit = 0.001,
            num = 50,
            duration_s=0.001,
            num_sim_per_rep=1,
            num_reps=1,
            node_drift_std=0.0,
            rx_noise_stds=0.0,
            mitigate_drift=True,
            drift_rate_std=0.0
    ):
        data_rows = []
        prediction_rows = []

        xs = np.linspace(ratio_limit, 1.0 - ratio_limit, num)

        for x in xs:
            delay_b = x * duration_s
            delay_a = duration_s-delay_b

            for i in range(num_reps):
                res, _ = sim(
                    num_exchanges=num_sim_per_rep,
                    resp_delay_s=(delay_b, delay_a),
                    node_drift_std=node_drift_std,
                    rx_noise_std=rx_noise_stds,
                    tx_delay_mean=0.0,
                    tx_delay_std=0.0, rx_delay_mean=0.0, rx_delay_std=0.0,
                    mitigate_drift=mitigate_drift,
                    drift_rate_std=drift_rate_std
                )
                tof_std = 1.0e09 * (res['est_tof_a']).std()
                tof_std_ci = logs.calc_ci_of_sd(tof_std, num_sim_per_rep, 0.01)
                tdoa_std = 1.0e09 * (res['est_tdoa']).std()
                tdoa_std_ci = logs.calc_ci_of_sd(tdoa_std, num_sim_per_rep, 0.01)

                data_rows.append(
                    {
                        'rdr': x,
                        'tof_std': tof_std,
                        'tof_std_ci_low': tof_std_ci[0],
                        'tof_std_ci_up': tof_std_ci[1],
                        'tof_ds_std': 1.0e09 * (res['est_tof_a_ds']).std(),
                        'tdoa_std': tdoa_std,
                        'tdoa_std_ci_low': tdoa_std_ci[0],
                        'tdoa_std_ci_up': tdoa_std_ci[1],
                        'tdoa_ds_std': 1.0e09 * (res['est_tdoa_ds']).std(),
                        'tdoa_half_cor_std': 1.0e09 * (res['est_tdoa_half_cor']).std(),
                        'tdoa_ds_half_cor_std': 1.0e09 * (res['est_tdoa_ds_half_cor']).std(),
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
                'predicted_tof_std_navratil': 1.0e09 * calc_predicted_tof_std_navratil(get_rx_noise('a', 'b'), get_rx_noise('b', 'a'), delay_b, delay_a),
                'predicted_tdoa_std': 1.0e09 * calc_predicted_tdoa_std(delay_b, delay_a)
            })
        return data_rows, prediction_rows

    for duration_s in [0.01]:

        data_rows, prediction_rows = proc_simulation_response_std(
            ratio_limit=ratio_limit,
            num=num,
            duration_s=duration_s,
            num_sim_per_rep=num_sim_per_rep,
            num_reps=1,
            node_drift_std=node_drift_std,
            rx_noise_stds=rx_noise_stds,
            mitigate_drift=mitigate_drift,
            drift_rate_std=0.0
        )

        data_df = pd.DataFrame(data_rows)
        pred_df = pd.DataFrame(prediction_rows)

        data_df = data_df.rename(columns={
            "tof_std": "Simulated DS-TWR SD",
            "tof_ds_std": "DS-TWR SD",
            "tdoa_std": "Simulated DS-TDoA SD",
            "tdoa_ds_std": "DS-TDoA SD",
            "tdoa_half_cor_std": "DS-TDoA (w/ DC) SD",
            "tdoa_ds_half_cor_std": "DS-TDoA DS (w/ DC) SD",
        })

        pred_df = pred_df.rename(columns={
            "predicted_tof_std": "Analytical DS-TWR SD",
            "predicted_tof_std_navratil": "Analytical DS-TWR SD\n[Navrátil and Vejražka]",
            "predicted_tdoa_std": "Analytical DS-TDoA SD"
        })

        ax = pred_df.plot.line(x='rdr', y=[
            'Analytical DS-TDoA SD',
            'Analytical DS-TWR SD\n[Navrátil and Vejražka]',
            'Analytical DS-TWR SD',
        ], alpha=1.0, color=['C2', 'gray', 'C4'], linestyle='--')

        ax = data_df.plot.line(x='rdr', ax=ax, y=[
            'Simulated DS-TDoA SD',
            'Simulated DS-TWR SD',
        ], alpha=0.5, color=['C2', 'C4'], linestyle='-')

        plt.fill_between(data_df['rdr'], data_df['tof_std_ci_low'], data_df['tof_std_ci_up'], color='C4', alpha=0.25)
        plt.fill_between(data_df['rdr'], data_df['tdoa_std_ci_low'], data_df['tdoa_std_ci_up'], color='C2', alpha=0.25)

        #data_df.plot.scatter(x='rdr', y='Simulated DS-TWR SD', ax=ax, c='C4', s=0.5, label='Simulated DS-TWR SD')
        #data_df.plot.scatter(x='rdr', y='Simulated DS-TDoA SD', ax=ax, c='C2', s=0.5, label='Simulated DS-TDoA SD')

        print("Mean", data_df['tof_mean'].mean(), data_df['tdoa_ds_mean'].mean())

        #ax.xaxis.set_major_formatter(lambda x, pos: r'$10^{{{}}}$'.format(int(round(x))))

        # def formatter(x):
        #     delay_b, = x * duration_s
        #     delay_a = duration_s - delay_b
        #
        #     delay_a = delay_a*1000
        #     delay_b = delay_b*1000
        #     return r'${{{}}}:{{{}}}$'.format(round(delay_b), round(delay_a))

        #ax.xaxis.set_major_formatter(lambda x, pos: formatter(x))

        #plt.axhline(y=np.sqrt(0.5), color='C0', linestyle='dotted', label = "Analytical ToF SD")
        #plt.axhline(y=np.sqrt(2.5), color='C1', linestyle='dotted', label = "Analytical TDoA SD")




        #plt.ylim([0.0, 1.0])
        #plt.xlim([-0.5, 0.5])
        #plt.ylim(0.2, 15)

        ax.set_axisbelow(True)
        ax.set_xlabel("Delay Ratio $\\ D_B:D_A$")
        ax.set_ylabel("Sample SD [ns]")


        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))

        #ax.xaxis.set_major_locator(MultipleLocator(1.0))

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

        plt.legend(ncol=2,handletextpad=0.2)
        plt.gcf().set_size_inches(6.15, 5.25)
        ticks = list(ax.get_yticks())
        labels = list(ax.get_yticklabels())

        #ticks.append(1.0e09 * np.sqrt((0.5*get_rx_noise('a', 'b'))**2 + (0.5*get_rx_noise('b', 'a'))**2))
        #labels.append(r'$\sqrt{0.5^2 \sigma_{BA}^2 + 0.5^2 \sigma_{AB}^2}$')

        ticks.append(np.sqrt(0.5))
        ticks.append(np.sqrt(2.5))
        ticks.append(np.sqrt(0.75))

        labels.append(r'$\sqrt{0.5}\sigma$')
        labels.append(r'$\sqrt{2.5}\sigma$')
        labels.append(r'$0.866\sigma$' "\n" r'$\approx \sqrt{0.75}'
                      r'\sigma$')

        ticks.append(np.sqrt(0.375))
        ticks.append(np.sqrt(1.875))
        labels.append(r'$\sqrt{0.375}\sigma$')
        labels.append(r'$\sqrt{1.875}\sigma$')

        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.125))

        print(ticks)
        print(labels)


        plt.tight_layout()

        save_and_crop("{}/tdoa_sim_rmse_reponse_delay_ratio_{}_{}.pdf".format(export_dir, round(duration_s*1000), 0), bbox_inches = 'tight', pad_inches = 0, crop=True)
        #plt.show()

        plt.close()


if __name__ == '__main__':
    config = load_env_config()
    load_plot_defaults()
    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']
    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])
    # TODO: Implement the logarithmic one again!
    export_tdoa_simulation_response_std(config['EXPORT_DIR'])




