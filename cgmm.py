#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.25

import math
import os
import numpy as np

LOG_PI = math.log(math.pi)

def gmm_posterior(obs, phi, sigma_inv, sigma_det):
    """
        This function returns log-posterior on GMM model G(x; 0, \phi * \sigma) given x
        for efficiency, do not calculate matrix invert and determinant inner function
        log G(x; \mu, \sigma) = -0.5 * D * log\pi - 0.5 * log |\sigma| - 0.5 * \
                (x - \mu) * \sigma^{-1} * (x - \mu)^T
        return complex type
    """
    dim = obs.size
    # transfer obs[vector] to matrix
    obs = np.matrix(obs)
    # exponent part, \mu = 0
    comp_e = obs * sigma_inv * obs.T / phi
    assert comp_e.size == 1
    # post = np.complex(-0.5 * (LOG_PI * dim + np.log(np.linalg.det(sigma)) + comp_e))
    post = np.complex(-0.5 * (LOG_PI * dim + np.log(sigma_det * (phi ** dim)) + comp_e))
    return post

def gmm_posterior_slow(obs, sigma):
    dim = obs.size
    obs = np.matrix(obs)
    comp_e = obs * sigma.I * obs.T
    post = np.complex(-0.5 * (LOG_PI * dim + np.log(np.linalg.det(sigma)) + comp_e))
    return post


class CGMM(object):
    def __init__(self, num_bins, time_steps, num_channels):
        """
            num_bins:   number of bins along frequent axis(usually 257)
            time_steps: number of frames per channel
            num_channels: number of channels, equals GMM dim
        """
        self.num_bins, self.time_steps = num_bins, time_steps
        self.dim = num_channels
        # lambda, phi, R for noisy/noise part
        self.lambda_ = np.zeros([num_bins, time_steps]).astype(np.complex)
        self.phi     = np.ones([num_bins, time_steps]).astype(np.complex)
        self.posterior = np.zeros([self.num_bins, self.time_steps]).astype(np.complex)

    def init_sigma(self, sigma):
        """
            Inputs: sigma is a np.matrix list 
            Keeps \sigma^{-1} and det(\sigma), \sigma equals \mean(y^H * y)
        """
        assert type(sigma) == list
        self.sigma_inv = [mat.I for mat in sigma]
        self.sigma_det = [np.linalg.det(mat) for mat in sigma]
        
    def covar_entropy(self):
        """
            Return entropy among eigenvalues of correlation matrix on 
            each frequency bin.
        """
        entropy = []
        for sigma_inv in self.sigma_inv:
            egval, _ = np.linalg.eig(sigma_inv.I)
            real_eigen = egval.real / egval.real.sum()
            entropy.append(-(real_eigen * np.log(real_eigen)).sum())
        return entropy

    def check_inputs(self, inputs):
        num_bins, time_steps, num_channels = inputs.shape
        assert num_bins == self.num_bins and time_steps == self.time_steps \
            and num_channels == self.dim, 'Inputs dim does not match CGMM config'

    # def log_likelihood(self, spectrums):
    #     self.check_inputs(spectrums)
    #     posteriors = 0.0
    #     for f in range(self.num_bins):
    #         for t in range(self.time_steps):
    #             posteriors += self.lambda_[f, t] * gmm_posterior(spectrums[f, t], \
    #                     self.phi[f, t], self.sigma_inv[f], self.sigma_det[f]) 
    #     return posteriors

    def accu_stats(self, spectrums):
        """
            Return posteriors on each frequency bin(size: F x T), in order to use
            them when updating lambda, we keep it as a class member
            We can get log_likelihood(function Q: eq.9) from posterior(by sum and average)
        """
        self.check_inputs(spectrums)
        # stats = np.zeros([self.num_bins, self.time_steps]).astype(np.complex)
        for f in range(self.num_bins):
            for t in range(self.time_steps):
                self.posterior[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t], \
                        self.sigma_inv[f], self.sigma_det[f]) 
        log_likelihood = (self.lambda_ * self.posterior).sum() / (self.num_bins * self.time_steps)
        return self.posterior, log_likelihood

    def update_lambda(self, spectrums, stats):
        """
            stats: sum of stats returned by function accu_stats
            update lambda: lambda = stats / \sum(stats) ref. eq.10
            Here using self.posterior calculated in function accu_stats to accelerate
            training progress.
        """
        print('update lambda...')
        assert stats.shape == self.posterior.shape
        # delete: avoid duplicated computation
        # for f in range(self.num_bins):
        #     for t in range(self.time_steps):
        #         self.lambda_[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t], \
        #                 self.sigma_inv[f], self.sigma_det[f])
        self.lambda_ = self.posterior / stats

    def update_phi(self, covar):
        """
            Update phi: ref. eq.9
        """
        print('update phi...')
        for f in range(self.num_bins):
            for t in range(self.time_steps):
                self.phi[f, t] = np.trace(covar[f * self.time_steps + t] * self.sigma_inv[f])
        self.phi = self.phi / self.dim

    def update_sigma(self, covar):
        """
            Update R: ref. eq.12
        """
        print('update sigma...')
        for f in range(self.num_bins):
            sum_lambda = self.lambda_[f].sum()
            R = np.matrix(np.zeros([self.dim, self.dim]).astype(np.complex))
            for t in range(self.time_steps):
                R += self.lambda_[f, t] * covar[f * self.time_steps + t] / self.phi[f, t]
            R = R / sum_lambda
            self.sigma_inv[f] = R.I 
            self.sigma_det[f] = np.linalg.det(R)

    def update_parameters(self, spectrums, covar, stats):
        """
            spectrums:  multi-channel training data(size: F x T x M)
            covar:      a python list, each item is a precomputed correlation matrix(y * y^H, 
                        type: np.matrix), we did it to avoid duplicate computing
            stats:      sum of stats in each CGMM part
        """
        self.check_inputs(spectrums)
        assert len(covar) == self.num_bins * self.time_steps and type(covar) == list
        self.update_lambda(spectrums, stats)
        self.update_phi(covar)
        self.update_sigma(covar)

class CGMMTrainer(object):
    def __init__(self, num_bins, time_steps, num_channels):
        self.noise_part = CGMM(num_bins, time_steps, num_channels)
        self.noisy_part = CGMM(num_bins, time_steps, num_channels)
        self.num_bins   = num_bins
        self.time_steps = time_steps

    def init_sigma(self, spectrums):
        """
            covar: precomputed correlation matrix of each channel
            Here we init noisy_part'R as correlation matrix of observed signal
        """
        print("initialize sigma...")
        num_bins, time_steps, num_channels = spectrums.shape
        self.covar = [y.H * y for y in [np.matrix(spectrums[f, t]) \
                for f in range(num_bins) for t in range(time_steps)]]
        self.noise_part.init_sigma([np.matrix(np.eye(num_channels, \
                num_channels).astype(np.complex)) for f in range(num_bins)])
        self.noisy_part.init_sigma([sum(self.covar[f * time_steps: \
              (f + 1) * time_steps]) / time_steps for f in range(num_bins)])
        
    # def log_likelihood(self, spectrums):
    #     return (self.noise_part.log_likelihood(spectrums) + \
    #             self.noisy_part.log_likelihood(spectrums)) / (self.num_bins * self.time_steps)

    def accu_stats(self, spectrums):
        print('accumulate statstics...')
        stats_y, post_y = self.noisy_part.accu_stats(spectrums)
        stats_n, post_n = self.noise_part.accu_stats(spectrums)
        return stats_y + stats_n, post_y + post_n
    
    def update_parameters(self, spectrums, stats):
        self.noise_part.update_parameters(spectrums, self.covar, stats)
        self.noisy_part.update_parameters(spectrums, self.covar, stats)

    def noise_lambda(self):
        e_n = self.noise_part.covar_entropy()
        e_y = self.noisy_part.covar_entropy()
        lambda_ = []
        for f in range(self.num_bins):
           lambda_.append(self.noise_part.lambda_[f] if e_n[f] > e_y[f] else self.noisy_part.lambda_[f])
        return np.array(lambda_)

    def save_param(self, dest):
        noise_lambda = self.noise_lambda()
        if not os.path.exists(dest):
            os.mkdir(dest)
        np.save(os.path.join(dest, 'noise_lambda'), noise_lambda)
        
    def train(self, spectrums, iters=30):
        self.init_sigma(spectrums)
        stats, likelihood = self.accu_stats(spectrums)
        for it in range(1, iters + 1):
            self.update_parameters(spectrums, stats)
            stats, likelihood = self.accu_stats(spectrums)
            print('epoch {0:2d}: Likelihood = ({1.real:.5f}, {1.imag:.5f}i)'.format(it, likelihood))

