import numpy as np


class AdaptiveProposal:
    def __init__(self, n_walkers, n_chains, n_dim,
                 target_acceptance_rate=0.234,
                 initial_variance=0.1):

        self.target_acceptance_rate = target_acceptance_rate

        variance = np.ones(shape=(n_walkers, n_chains, n_dim, n_dim))
        variance = variance * np.expand_dims(np.arange(1, n_chains + 1) / n_chains * 10, axis=(0, -2, -1))
        variance = variance * np.eye(n_dim, n_dim)  # Make variables independent initially
        variance *= initial_variance
        self.L_variance = np.linalg.cholesky(variance)

        self.params_shape = (n_walkers, n_chains, n_dim)
        self.initial_variance = initial_variance

        self.move = 0
        self.covariances = []
        self.init()

    def init(self):
        pass

    def __call__(self, prev_state=None):
        shape = self.params_shape
        # Perform multivariate batch sampling

        samples = np.random.normal(size=np.prod(shape))
        samples = samples.reshape(shape)
        L = self.L_variance
        move = np.squeeze(L @ np.expand_dims(samples, axis=-1))
        self.move = move

        if prev_state is None:
            state = move
        else:
            state = np.array(prev_state)
            state = state + move

        return state

    def update_proposal(self, parameters, priors, likelihoods, step_accepts, alpha, iN):
        raise Exception("Not implemented")


class RAMProposal(AdaptiveProposal):

    def init(self):
        c = self.params_shape[2]  # Chosen to be n_dim according to Miasojedow et al. In the range (0, 1]
        e = 0.6  # In the range (0.5, 1)
        self.nu = lambda n: min((0.9, c * (n + 1) ** (-e)))

    def update_proposal(self, parameters, priors, likelihoods, step_accepts, alpha, iN):
        if iN == 0:
            return
        L = self.L_variance

        # U = self.move  #
        U = parameters[iN] - parameters[iN - 1]  # if iN > 0 else parameters[iN]
        M = np.expand_dims(U, 3) @ np.expand_dims(U, 2)
        m = np.power(np.linalg.norm(U), 2)
        M = M / m if m != 0.0 else np.zeros(shape=M.shape)
        I = np.expand_dims(np.eye(L.shape[3]), (0, 1))
        COV = L @ (I + self.nu(iN) * np.expand_dims(alpha - self.target_acceptance_rate,
                                                    (2, 3)) * M) @ np.transpose(L, axes=(0, 1, 3, 2))

        # print(COV[0][0])
        self.L_variance = np.linalg.cholesky(COV)

        self.covariances.append(COV)


class GeneralizedAdaptiveProposal(AdaptiveProposal):

    def init(self):
        n_walkers, n_chains, n_dim = self.params_shape

        radius = self.initial_variance * np.ones(shape=(n_walkers, n_chains))  # Single radius for each chain

        mean = np.zeros(shape=(n_walkers, n_chains, n_dim))

        variance = np.ones(shape=(n_walkers, n_chains, n_dim, n_dim))
        # variance = variance * np.expand_dims(np.arange(1, n_chains + 1) / n_chains * 10, axis=(0, -2, -1))
        variance = variance * np.eye(n_dim, n_dim)  # Make variables independent initially

        self.mean = None  # mean
        self.radius = radius
        self.variance = variance

        self.L_variance = np.linalg.cholesky(variance)
        self.nu = lambda n: min(0.02, ((n + 1) / 5) ** (-1))
        self.radii = []

    def update_proposal(self, parameters, priors, likelihoods, step_accepts, alpha, iN):
        if self.mean is None:
            self.mean = parameters[iN]

        cur_params = parameters[iN]
        diff = cur_params - self.mean
        radius = self.radius * np.exp(self.nu(iN) * (alpha - self.target_acceptance_rate))
        mean = self.mean + self.nu(iN) * diff
        CORR = np.expand_dims(diff, axis=3) @ np.expand_dims(diff, axis=2)
        COV = self.variance + self.nu(iN) * (CORR - self.variance)

        try:
            self.L_variance = np.linalg.cholesky(np.expand_dims(radius, (2, 3)) * COV)
        except:
            print("Covariance wasn't positive definite.")
            diag = np.diagonal(COV, axis1=2, axis2=3)
            R = np.sum(np.abs(COV), axis=2) - np.abs(diag)
            lower_bounds = diag - R
            COV += (np.expand_dims(np.abs(R) * 1.1 * (lower_bounds <= 0), axis=-1) * np.eye(16, 16))

            self.L_variance = np.linalg.cholesky(np.expand_dims(radius, (2, 3)) * COV)

        self.radius = radius
        self.mean = mean
        self.variance = COV

        self.radii.append(radius)
        self.covariances.append(COV)
