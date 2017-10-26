#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.25

import math
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
        self.lambda_ = np.random.rand(num_bins, time_steps).astype(np.complex)
        self.phi     = np.ones([num_bins, time_steps]).astype(np.complex)
        # type matrix
        self.R       = [np.matrix(np.eye(num_channels, num_channels).astype(np.complex)) \
                            for i in range(num_bins)] 
    
    def check_inputs(self, inputs):
        num_bins, time_steps, num_channels = inputs.shape
        assert num_bins == self.num_bins and time_steps == self.time_steps \
            and num_channels == self.dim, 'inputs dim does not match CGMM config'

    def log_likelihood(self, spectrums):
        self.check_inputs(spectrums)
        posteriors = 0.0
        for f in range(self.num_bins):
            sigma_inv = self.R[f].I
            sigma_det = np.linalg.det(self.R[f])
            for t in range(self.time_steps):
                posteriors += self.lambda_[f, t] * gmm_posterior(spectrums[f, t], \
                        self.phi[f, t], sigma_inv, sigma_det) 
        return posteriors

    def accu_stats(self, spectrums):
        self.check_inputs(spectrums)
        stats = np.zeros([self.num_bins, self.time_steps]).astype(np.complex)
        for f in range(self.num_bins):
            sigma_inv = self.R[f].I
            sigma_det = np.linalg.det(self.R[f])
            for t in range(self.time_steps):
                stats[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t], \
                        sigma_inv, sigma_det) 
        return stats

    def update_lambda(self, spectrums, stats):
        print('update lambda...')
        for f in range(self.num_bins):
            sigma_inv = self.R[f].I
            sigma_det = np.linalg.det(self.R[f])
            for t in range(self.time_steps):
                self.lambda_[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t], \
                        sigma_inv, sigma_det) / stats[f, t]

    def update_phi(self, spectrums):
        print('update phi...')
        for f in range(self.num_bins):
            inv_R = self.R[f].I
            for t in range(self.time_steps):
                y = np.matrix(spectrums[f, t])
                self.phi[f, t] = np.trace(y.H * y * inv_R) / self.dim

    def update_R(self, spectrums):
        print('update R...')
        for f in range(self.num_bins):
            sum_lambda = self.lambda_[f].sum()
            self.R[f] = 0
            for t in range(self.time_steps):
                y = np.matrix(spectrums[f, t])
                self.R[f] += self.lambda_[f, t] * y.H * y / self.phi[f, t]
            self.R[f] = self.R[f] / sum_lambda

    def update_parameters(self, spectrums, stats):
        self.check_inputs(spectrums)
        self.update_lambda(spectrums, stats)
        self.update_phi(spectrums)
        self.update_R(spectrums)

class CGMMTrainer(object):
    def __init__(self, num_bins, time_steps, num_channels):
        self.noise_part = CGMM(num_bins, time_steps, num_channels)
        self.noisy_part = CGMM(num_bins, time_steps, num_channels)
        self.num_samples = num_bins * time_steps
    
    def log_likelihood(self, spectrums):
        return (self.noise_part.log_likelihood(spectrums) + \
                self.noisy_part.log_likelihood(spectrums)) / self.num_samples

    def accu_stats(self, spectrums):
        print('accumulate statstics...')
        return self.noisy_part.accu_stats(spectrums) + \
                self.noise_part.accu_stats(spectrums)
    
    def update_parameters(self, spectrums, stats):
        self.noise_part.update_parameters(spectrums, stats)
        self.noisy_part.update_parameters(spectrums, stats)

    def train(self, spectrums, iters=30):
        print('Likelihood: {:.4}'.format(self.log_likelihood(spectrums)))
        for it in range(1, iters + 1):
            stats = self.accu_stats(spectrums)
            self.update_parameters(spectrums, stats)
            print('epoch {:2d}: Likelihood = {:.4}'.format(it, \
                    self.log_likelihood(spectrums)))

