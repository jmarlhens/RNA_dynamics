import numpy as np
from numpy.fft import fft, ifft


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


def convergence_test(samples):
    """
    Checks for mixing and stationarity by employing multiple walkers following Gelman et al. (Bayesian Data Analysis Third edition (with errors ﬁxed as of 15 February 2021), Gelman Rubin approach).
    Gives rise to the potential scale reduction with narrows 1 for n -> \infty. Large values indicate, that the number of samples should be increased.

    Value below 1.2 favorable
    :param samples:
    :return:
    """

    # We only consider the first chain here, as this is the chain sampling from the untempered posterior
    samples = samples[:, :, 0]

    # Drop burn in phase
    data = samples[int(len(samples) / 2):]
    length = len(data)
    n = int(length / 2)
    split = data[:n], data[n:2 * n]
    split = np.concatenate(split, axis=1)

    # Computing the between chain variance B
    per_chain_average = np.mean(split, axis=0)
    B = n * np.var(per_chain_average)

    # Computing the within chain variance W
    per_chain_variance = np.var(split, axis=0)
    W = np.mean(per_chain_variance)

    var_plus = ((n - 1) * W + B) / n

    R_hat = np.sqrt(var_plus / W)
    return R_hat


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
