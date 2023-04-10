import os
import progressbar
import numpy as np

import matplotlib.pyplot as plt
from utility import slugify, cached, init_cache, load_env_config

import pandas as pd

METHOD_PREFIX = 'export_'

CONFIDENCE_FILL_COLOR = '0.8'
PERCENTILES_FILL_COLOR = '0.5'
COLOR_MAP = 'tab10'

c_in_air = 299702547.236


def load_plot_defaults():
    # Configure as needed
    plt.rc('lines', linewidth=2.0)
    plt.rc('legend', framealpha=1.0, fancybox=True)
    plt.rc('errorbar', capsize=3)
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)
    plt.rc('font', size=11)


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
    plt.ylim(0.0, 12.0)
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

    plt.grid(color='gray', linestyle='dashed')

    plt.tight_layout()

    plt.savefig("{}/sim_rmse.pdf".format(export_dir))
    #plt.show()

    plt.close()

#
#
# def export_sine_wave_example(config, export_dir):
#     # We got multiple experiment runs with individual measurements
#     num_runs = 10
#
#     # Create our measurement steps
#     xs = np.linspace(0, 2 * np.pi, 100, endpoint=True)
#
#     # We also collect overall data for mean and confidence interval
#     overall_data = []
#
#     for r in range(0, num_runs):
#         name = "Sine Wave Run {}".format(r)
#
#         def proc():
#             # you can load your data from a database or CSV file here
#             # we will randomly generate data
#             ys = np.sin(np.array(xs))
#             # we add some uniform errors
#             ys += np.random.uniform(-0.1, 0.1, len(xs))
#             return ys
#
#         # If caching is enabled, this line checks for available cache data
#         # If no data was found, the proc callback is executed and the result cached
#         # Use ys = proc() if caching not yet wanted
#         ys = cached(('sine_wave', r), proc)
#
#         # We also add the data to overall_data
#         overall_data.append(ys)
#
#         plt.clf()
#
#         # Plot the main data
#         plt.plot(xs, ys, linestyle='-', label="Sin Run {}".format(r), color='C' + str(r + 1))
#
#         plt.legend()
#         plt.xlabel("x")
#         plt.ylabel("sin(x)")
#         plt.axis([None, None, None, None])
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(export_dir + slugify(name) + ".pdf", format="pdf")
#         plt.close()
#
#     overall_data = np.array(overall_data)
#
#     # We swap the axes to get all values at the first position together
#     overall_data = np.swapaxes(overall_data, 0, 1)
#
#     # We can then merge each step to get the mean
#     mean = np.mean(overall_data, axis=1)
#
#     # calculate confidence intervals for each mean
#     cis = 1.96 * np.std(overall_data, axis=1) / np.sqrt(np.size(overall_data, axis=1))
#
#     # Calculate the lower and upper bounds of the 95% percentiles
#     # This describes that 95% of the measurements (for each timestep) are within that range
#     # Use standard error to determine the "quality" of your calculated mean
#     (lq, uq) = np.percentile(overall_data, [2.5, 97.5], axis=1)
#
#     # clear the plot
#     plt.clf()
#
#     # plot the mean values
#     plt.plot(xs, mean, linestyle='-', label="Mean", color='C1')
#
#     # plot the confidence interval for the computed means
#     plt.fill_between(xs, (mean - cis), (mean + cis), color=CONFIDENCE_FILL_COLOR, label='CI')
#
#     # plot also the 95% percentiles, i.e., the range in that 95% of our data falls, this is quite different from the confidence interval
#     plt.fill_between(xs, lq, uq, color=PERCENTILES_FILL_COLOR, label='95% Percentiles')
#
#     plt.legend()
#     plt.xlabel("x")
#     plt.ylabel("sin(x)")
#     plt.axis([None, None, None, None])
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(export_dir + slugify("Sine Wave Mean") + ".pdf", format="pdf")
#     plt.close()
#
#
# def export_bar_example(config, export_dir):
#     # we want to display two bar grahps
#     # see export_sine_wave_example
#
#     num_data_points = 100
#
#     data_a = cached(('example_2', 'a'), lambda: np.random.uniform(75, 82, num_data_points))
#     data_b = cached(('example_2', 'b'), lambda: np.random.uniform(70, 96, num_data_points))
#
#     mean_a = np.mean(data_a)
#     mean_b = np.mean(data_b)
#
#     std_a = np.std(data_a)
#     std_b = np.std(data_b)
#
#     plt.clf()
#
#     fig, ax = plt.subplots()
#
#     ax.bar(["Interesting Bar A", "Somewhat Nice Bar B"], [mean_a, mean_b], yerr=[std_a, std_b], align='center',
#            ecolor='black', capsize=5, color=['C1', 'C2', 'C3'])
#     ax.yaxis.grid(True)
#     plt.ylabel("Something [unit]")
#     plt.axis([None, None, 0, 100])
#
#     # Adapt the figure size as needed
#     fig.set_size_inches(5.0, 8.0)
#     plt.tight_layout()
#     plt.savefig(export_dir + slugify(("Bar", 5.0, 8.0)) + ".pdf", format="pdf")
#
#     fig.set_size_inches(4.0, 4.0)
#     plt.tight_layout()
#     plt.savefig(export_dir + slugify(("Bar", 4.0, 4.0)) + ".pdf", format="pdf")
#
#     plt.close()



if __name__ == '__main__':

    config = load_env_config()

    load_plot_defaults()

    assert 'EXPORT_DIR' in config and config['EXPORT_DIR']

    if 'CACHE_DIR' in config and config['CACHE_DIR']:
        init_cache(config['CACHE_DIR'])

    steps = [
        export_simulation_performance,
    ]

    for step in progressbar.progressbar(steps, redirect_stdout=True):
        name = step.__name__.removeprefix(METHOD_PREFIX)
        print("Handling {}".format(name))
        export_dir = os.path.join(config['EXPORT_DIR'], name) + '/'
        os.makedirs(export_dir, exist_ok=True)
        step(config, export_dir)
