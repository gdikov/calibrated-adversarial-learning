import numpy as np
from scipy import stats

_EPSILON = 1e-11


class Bifurcation:
    """
    Data factory for a fork-like bimodal function of the form:

   0.5 |    -------\
       |             \
   0.0 |              -----
       |             /
   -0.5|    -------/
       |----------------------
           0     0.4  0.8  1

    A formal definition is given in Eq. (8) in the main text.
    """
    def __init__(self, pi=0.0, sigma=0.01):
        self.noise_std = sigma
        self.mix_coef = np.array([pi, 1 - pi])

    @staticmethod
    def _split_intervals(xs):
        # first interval: 0 ≤ x < 0.4
        interval1 = xs < 0.4
        # second interval: 0.4 ≤ x < 0.8
        interval2 = np.logical_and(xs >= 0.4, xs < 0.8)
        # third interval: 0.8 ≤ x < 1
        interval3 = xs >= 0.8
        return interval1, interval2, interval3

    def produce(self, batch_size, seed=None):
        """Generate infinite data."""
        rng = np.random.RandomState(seed)
        while True:
            xs = rng.uniform(0, 1, size=batch_size)
            interval1, interval2, interval3 = self._split_intervals(xs)
            xs1 = xs[interval1]
            ys1 = rng.choice(
                [0.5, -0.5],
                p=self.mix_coef,
                size=len(xs1)
            ) + rng.standard_normal(len(xs1)) * self.noise_std
            xs2 = xs[interval2]
            mask_modes = rng.choice(
                [True, False], p=self.mix_coef, size=len(xs2)
            )
            fx1 = -0.5 / 0.4 * xs2[mask_modes] + 1.0
            fx2 = 0.5 / 0.4 * xs2[~mask_modes] - 1.0
            xs2 = np.concatenate([xs2[mask_modes], xs2[~mask_modes]], axis=0)
            ys2 = np.concatenate([fx1, fx2], axis=0) + rng.standard_normal(
                len(xs2)) * self.noise_std
            xs3 = xs[xs >= 0.8]
            ys3 = rng.standard_normal(len(xs3)) * self.noise_std
            # concatenate all samples
            ys = np.concatenate([ys1, ys2, ys3], axis=0).astype(np.float32)
            xs = np.concatenate([xs1, xs2, xs3], axis=0).astype(np.float32)
            yield xs, ys

    def ll(self, xs, ys):
        """Return the log-likelihood for a given (`x`, `y`) pair.
        Exact log-likelihood can be computed since the density,
        for a fixed `x`, is a mixture of two Gaussians.
        """
        interval1, interval2, interval3 = self._split_intervals(xs)
        ll1 = np.log(
            self.mix_coef[0] * stats.norm.pdf(ys[interval1], loc=0.5, scale=self.noise_std)
            + self.mix_coef[1] * stats.norm.pdf(ys[interval1], loc=-0.5, scale=self.noise_std)
            + _EPSILON
        )
        loc_u = -0.5 / 0.4 * xs[interval2] + 1.0
        loc_l = 0.5 / 0.4 * xs[interval2] - 1.0
        ll2 = np.log(
            self.mix_coef[0] * stats.norm.pdf(ys[interval2], loc=loc_u, scale=self.noise_std)
            + self.mix_coef[1] * stats.norm.pdf(ys[interval2], loc=loc_l, scale=self.noise_std)
            + _EPSILON
        )
        ll3 = np.log(
            stats.norm.pdf(ys[interval3], loc=0, scale=self.noise_std)
            + _EPSILON
        )
        # combine all intervals
        ll_all = np.zeros_like(ys)
        ll_all[interval1] = ll1
        ll_all[interval2] = ll2
        ll_all[interval3] = ll3
        return ll_all

    def mean(self, xs):
        """Return the mean `y` for a given `x`."""
        interval1, interval2, _ = self._split_intervals(xs)
        mean1 = 0.5 * self.mix_coef[0] - 0.5 * self.mix_coef[1]
        mask2 = np.logical_and(xs >= 0.4, xs < 0.8)
        loc_u = -0.5 / 0.4 * xs[mask2] + 1.0
        loc_l = 0.5 / 0.4 * xs[mask2] - 1.0
        mean2 = loc_u * self.mix_coef[0] + loc_l * self.mix_coef[1]
        # combine all intervals and skip the 0.8 ≤ x < 1
        # as it is already zero-centered
        means = np.zeros_like(xs)
        means[interval1] = mean1
        means[interval2] = mean2
        return means
