import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, animation
from matplotlib.colors import LinearSegmentedColormap
from numpy.fft import fft, ifft
from matplotlib.backends.backend_pdf import PdfPages
from numpy.ma.core import squeeze

COLORS = [(0.30, 0.56, 1.00), (0.35, 0.24, 1.00), (1.00, 0.00, 0.42), (1.00, 0.40, 0.10), (1.00, 0.65, 0.19)]
COLORS_DARK = [(0.00, 0.24, 0.90), (0.28, 0.00, 0.84), (0.80, 0.00, 0.27), (0.90, 0.24, 0.00), (1.00, 0.50, 0.00)]
COLORS_MAIN = [(0.30, 0.55, 1.00), (1.00, 0.40, 0.10)]
COLORS_CMAP_ORANGE = ["#FFFFFF", "#FF5500", "#B3003C"]
COLORS_CMAP_BLUE = ["#FFFFFF", "#69A3FF", "#4400D6"]

COLORMAP_ORANGE = LinearSegmentedColormap.from_list("my_cmap", COLORS_CMAP_ORANGE)
COLORMAP_BLUE = LinearSegmentedColormap.from_list("my_cmap", COLORS_CMAP_BLUE)

COLOR_GRAY = "#808080"
COLOR_REPLICATES = COLORS_DARK[2:]  # Use only redish colors for replicates


# © Copyright 2012-2021, Dan Foreman-Mackey & contributors.
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


# © Copyright 2012-2021, Dan Foreman-Mackey & contributors.
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# © Copyright 2012-2021, Dan Foreman-Mackey & contributors.
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


# © Copyright 2012-2021, Dan Foreman-Mackey & contributors.
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
def autocorr_new(y, c=5.0):
    """
    Computes autocorrelation time of ensemble of walkers by computing the individual autocorrelation times first and then averaging these
    :param y: The samples of the ensemble (first dimension is over walkers, second over time)
    :param c:
    :return:
    """
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def correlation(X, Y, norm=True):
    """
    :param X:
    :param Y:
    :param norm:
    :return: Correlation vector giving rise to correlation between variables of X to the one of Y position wise
    """

    # X = np.expand_dims(X, axis=1)
    # Y = np.expand_dims(Y, axis=1)
    # X_mean = np.mean(X, axis=0)
    # Y_mean = np.mean(Y, axis=0)
    # X_ft = fft(X - X_mean, axis=0)
    # Y_ft = fft(np.transpose(Y - Y_mean, axes=(0, 1)), axis=1)
    # outer_product =
    # Next step is to perform inverse FFT of matrice

    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_ft = fft(X - X_mean, axis=0)
    Y_ft = fft(Y - Y_mean, axis=0)
    inner_product = X_ft * np.conjugate(Y_ft)
    corr = ifft(inner_product, axis=0).real

    if norm:
        corr /= corr[0]

    return corr


# Implementation follows automatic windowing procedure of Sokal
def autowindow(taus, c):
    cond = np.expand_dims(np.arange(len(taus)), axis=list(np.arange(1, len(taus.shape[1:]) + 1))) >= c * taus
    M = np.argmax(cond, axis=0)
    return M


def integrated_autocorrelation_time(X, c=6):
    auto_corr = correlation(X, X)
    taus = 2 * np.cumsum(auto_corr, axis=0) - 1 * auto_corr[0]
    # As auto_corr[1]=auto_corr[-1], the factor 2 makes the cumsum equal to a symmetric sum (e.g. auto_corr[1] + auto_corr[-1] = 2* auto_corr[1]).
    # However, we have to correct for 2*auto_corr[0], wherefore we substract 1 * auto_corr[0] (in the case of normalized values, also -1 is applicable instead).

    M = autowindow(taus, c=c)
    dims = np.meshgrid(*[np.arange(elem) for elem in M.shape],
                       indexing='ij'  # Use 'ij' to get arrays aligned with M's dimensions
                       )
    tau = taus[M, *dims]
    return tau


def convergence_test(samples, per_parameter_test=False):
    """
    Checks for mixing and stationarity by employing multiple walkers following Gelman et al. (Bayesian Data Analysis Third edition (with errors ﬁxed as of 15 February 2021), Gelman Rubin approach).
    Gives rise to the potential scale reduction with narrows 1 for n -> \infty. Large values indicate, that the number of samples should be increased.

    Value below 1.2 favorable
    :param samples:
    :return:
    """

    axis = 0 if per_parameter_test else None

    # We only consider the first chain here, as this is the chain sampling from the untempered posterior
    samples = samples[:, :, 0]
    # Burn in phase already dropped before hand
    # # Drop burn in phase
    # data = samples[int(len(samples) / 2):]
    data = samples
    length = len(data)
    n = int(length / 2)
    split = data[:n], data[n:2 * n]
    split = np.concatenate(split, axis=1)

    # Computing the between chain variance B
    per_chain_average = np.mean(split, axis=0)

    B = n * np.var(per_chain_average, axis=axis)

    # Computing the within chain variance W
    per_chain_variance = np.var(split, axis=0)
    W = np.mean(per_chain_variance, axis=axis)

    var_plus = ((n - 1) * W + B) / n

    R_hat = np.sqrt(var_plus / W)
    return R_hat


def animate_parameter_trace_2D(data):
    # Code created with the help of Perplexity
    data = np.array(data)
    if len(data.shape) == 2:
        data = np.expand_dims(data, 1)

    n_w = data.shape[1]
    scatter_plots = []
    line_plots = []

    def update(version):
        cur_data = data[:version]
        pass
        # Line connects the mean trajectory up to the current version
        if version > 0 and len(cur_data) > 0:
            # print(f"I'm here with version {version} and shape {cur_data.shape}")
            # print(cur_data[:, 0])
            for iW in range(n_w):
                scatter_plots[iW].set_offsets(cur_data[:, iW])
                line_plots[iW].set_data(cur_data[:, iW, 0], cur_data[:, iW, 1])
        else:
            for iW in range(n_w):
                line_plots[iW].set_data([], [])

        pass
        ax.set_title(f'Numpy Array Evolution - Version {version + 1}', fontsize=16, weight='bold')
        return scat, line

    fig, ax = plt.subplots(figsize=(8, 5))

    for iW in range(n_w):
        scat = ax.scatter([], [], s=60, color=COLORS[iW % 5], zorder=2, alpha=0.6)
        line, = ax.plot([], [], "--", lw=2, color=COLORS[iW % 5], zorder=1, alpha=0.4)
        scatter_plots.append(scat)
        line_plots.append(line)

    ax.set_xlim(np.min(data[:, :, 0]) - 0.5, np.max(data[:, :, 0]) + 0.5)
    ax.set_ylim(np.min(data[:, :, 1]) - 0.5, np.max(data[:, :, 1]) + 0.5)
    ax.set_xlabel('Element Index', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Numpy Array Evolution Over Versions', fontsize=16, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=0.01, blit=True)

    FFwriter = animation.FFMpegWriter(fps=30)
    ani.save('animation.mp4', writer=FFwriter, dpi=180)
    # ani.save('array_evolution.gif', writer='pillow', dpi=180)  # Save as GIF
    plt.close(fig)
    print(f"Wrote animation to {'array_evolution.gif'}")


def sliding_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_traces(data, file_path, param_names=[], N=1000, add_histogram=True):
    # Code created with the help of Perplexity

    data = np.array(data)
    num_plots = data.shape[3]
    num_chains = data.shape[2]
    num_walkers = data.shape[1]
    num_samples = data.shape[0]
    X = np.arange(num_samples)
    with PdfPages(file_path) as pdf:
        plots_per_page = 3
        num_pages = int(np.ceil(num_plots / plots_per_page))

        for page in range(num_pages):
            ncols = 2 if add_histogram else 1
            # height_ratios = [1, 1, 1]
            width_ratios = [5, 1] if add_histogram else [1]
            fig, axes = plt.subplots(nrows=plots_per_page, ncols=ncols, figsize=(15, 10), squeeze=False, width_ratios=width_ratios)  # Landscape mode

            for i in range(plots_per_page):
                plot_idx = page * plots_per_page + i
                if plot_idx < num_plots:
                    ax = axes[i, 0]
                    for iW in range(num_walkers):
                        ax.plot(X, data[:, iW, 0, plot_idx], color=COLORS[iW], lw=1, zorder=2, alpha=0.8)

                        sliding_window = [data[iX: iX + N + 1, iW, 0, plot_idx] for iX in range(data.shape[0] - N)]
                        mean = np.array(list(map(np.mean, sliding_window)))
                        std_dev = np.array(list(map(np.std, sliding_window)))
                        ax.plot(X[N:], mean, "--", color=COLORS[iW], lw=1, alpha=0.4, zorder=1)  # COLORS_MAIN[0])
                        ax.fill_between(X[N:], mean - std_dev, mean + std_dev, color=COLORS[iW], zorder=0, alpha=0.2)

                    if len(param_names) > plot_idx:
                        ax.set_title(f'{param_names[plot_idx]}', fontsize=14)
                    else:
                        ax.set_title(f'Parameter {plot_idx}', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("Iteration")
                    ylim = ax.get_ylim()
                    if add_histogram:
                        ax = axes[i, 1]
                        bins = np.linspace(ylim[0], ylim[1], 100)
                        n_samples = data.shape[0]
                        offset = n_samples // 2
                        # Prior Hist
                        ax.hist(data[offset:, :, -1, plot_idx].flatten(), bins=bins, color="k",
                                orientation="horizontal", alpha=0.5)

                        for iW in range(num_walkers):
                            ax.hist(data[offset:, iW, 0, plot_idx].flatten(), bins=bins, color=COLORS[iW], orientation="horizontal", alpha=0.5)



                else:
                    # Hide unused subplots
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved plots file to '{file_path}'.", flush=True)




class MCMCResultsWriter:

    file = None

    def __init__(self, path, param_names=None):
        abspath = os.path.abspath(path)
        if self.file is not None and os.path.abspath(self.file.name) != abspath:
            self.close()

        self.file = open(abspath, "a")
        print(f"Opened file {self.file.name}")


        if param_names is None:
            param_names = [f"Parameter {iX}" for iX in range(self.n_dim)]
        self._write_to_file(
            lines=["iteration,walker,chain," + ",".join(param_names) + ",likelihood,prior,posterior,step_accepted\n"])
        self.start_index = 0

    def _write_to_file(self, lines):
        self.file.writelines(lines)
        self.file.flush()
        print(f"Updated file {self.file.name}")

    def save_state_in_file(self, parameters, priors, likelihoods, step_accepts, swap_accepts, index=None):
        start_index = self.start_index
        end_index = parameters.shape[0] if index is None else index + 1

        n_walkers = parameters.shape[1]
        n_chains = parameters.shape[2]
        n_dim = parameters.shape[3]

        cur_parameters = parameters[start_index:end_index]
        cur_likelihoods = likelihoods[start_index:end_index]
        cur_priors = priors[start_index:end_index]
        cur_step_accepts = step_accepts[start_index:end_index]

        iterations = np.arange(start_index, end_index)
        walkers = np.arange(n_walkers)
        chains = np.arange(n_chains)


        # Create meshgrid for all combinations
        iter_grid, walker_grid, chain_grid = np.meshgrid(
            iterations, walkers, chains, indexing="ij"
        )

        cols = []
        cols += [iter_grid.flatten(),
                 walker_grid.flatten(),
                 chain_grid.flatten()]
        cols += [cur_parameters[..., iP].flatten() for iP in range(n_dim)]
        cols += [cur_likelihoods.flatten(),
                 cur_priors.flatten(),
                 (cur_priors.flatten() + cur_likelihoods.flatten()),
                 cur_step_accepts.flatten()]

        data = np.concatenate([np.expand_dims(col, axis=1) for col in cols], axis=1)

        encoded_data = list(map(lambda row: ",".join(row.astype(str)) + "\n", data))

        self._write_to_file(lines=encoded_data)
        # if end_index - start_index >= 1:
        if end_index is not None:
            self.start_index = end_index

    def close(self):
        if self.file is not None:
            self.file.close()
            print(f"Closed file {self.file.name}")
            self.file = None


def load_state_from_file(path):
    abspath = os.path.abspath(path)

    swap_accepts = None

    df = pd.read_csv(abspath)

    data = df.values
    n_samples = int(np.max(data[:,0])) + 1
    n_walkers = int(np.max(data[:, 1])) + 1
    n_chains = int(np.max(data[:, 2])) + 1

    data = data.reshape((n_samples, n_walkers, n_chains, -1))


    parameters= data[..., 3:data.shape[-1] - 4]
    likelihoods = data[..., -4]
    priors = data[..., -3]
    posterior = data[..., -2]
    step_accepts = data[..., -1]
    index = n_samples - 1

    return parameters, priors, likelihoods, step_accepts, swap_accepts, index


if __name__ == '__main__':
    X = np.random.uniform(size=(10, 1))
    Y = np.random.uniform(size=(10, 1))
    X_c = X - np.mean(X)
    Y_c = Y - np.mean(Y)

    #
    # c1 = np.array([np.correlate(X_c, np.roll(Y_c, iX))[0] for iX in range(len(X))])
    # c1 = c1 / c1[0]
    # c2 = correlation(X, Y)

    # X = np.repeat(X, 3)
    # X = np.random.normal(X, 1)

    tau = integrated_autocorrelation_time(X, c=6)
    tau_2 = autocorr_new(X.transpose(), c=6)
    pass
