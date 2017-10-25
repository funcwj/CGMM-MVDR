#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.25

import math
import numpy as np

LOG_PI = math.log(math.pi)

def gmm_posterior(obs, sigma):
    """
        log G(x; \mu, \sigma) = -0.5 * D * log\pi - 0.5 * log |\sigma| - 0.5 * \
                (x - \mu) * \sigma^{-1} * (x - \mu)^T
    """
    dim = obs.size
    inv_conv = np.linalg.inv(sigma)
    # exponent part, \mu = 0
    comp_e   = np.matmul(np.matmul(obs, inv_conv), obs.T).real
    post = -0.5 * (LOG_PI * dim + np.log(np.abs(np.linalg.det(sigma))) + comp_e)  
    return post

class CGMM(object):
    def __init__(self, num_bins, time_steps, num_channels):
        self.num_bins, self.time_steps = num_bins, time_steps
        self.dim = num_channels
        # lambda, phi, R for noisy/noise part
        self.lambda_ = np.random.rand(num_bins, time_steps)
        self.phi     = np.random.rand(num_bins, time_steps)
        self.R       = [np.eye(num_channels, num_channels).astype(np.complex) for i in range(num_bins)] 
    
    def check_inputs(self, inputs):
        num_bins, time_steps, num_channels = inputs.shape
        assert num_bins == self.num_bins and time_steps == self.time_steps \
            and num_channels == self.dim, 'inputs dim does not match CGMM config'

    def log_likelihood(self, spectrums):
        self.check_inputs(spectrums)
        posteriors = 0.0
        for f in range(self.num_bins):
            for t in range(self.time_steps):
                posteriors += self.lambda_[f, t] * gmm_posterior(spectrums[f, t], self.phi[f, t] * self.R[f]) 
        return posteriors

    def accu_stats(self, spectrums):
        self.check_inputs(spectrums)
        stats = np.zeros([self.num_bins, self.time_steps])
        for f in range(self.num_bins):
            for t in range(self.time_steps):
                stats[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t] * self.R[f]) 
        return stats

    def update_lambda(self, spectrums, stats):
        for f in range(self.num_bins):
            for t in range(self.time_steps):
                self.lambda_[f, t] = gmm_posterior(spectrums[f, t], self.phi[f, t] * self.R[f]) / stats[f, t]

    def update_phi(self, spectrums):
        for f in range(self.num_bins):
            inv_R = np.matrix(np.linalg.inv(self.R[f]))
            for t in range(self.time_steps):
                y = np.matrix(spectrums[f, t])
                self.phi[f, t] = np.trace(y.H * y * inv_R) / self.dim

    def update_R(self, spectrums):
        for f in range(self.num_bins):
            sum_lambda = self.lambda_[f].sum()
            stats_R = np.zeros([self.dim, self.dim])
            for t in range(self.time_steps):
                y = np.matrix(spectrums[f, t])
                stats_R += self.lambda_[f, t] * y.H * y / self.phi[f, t]
            self.R[f] = stats_R

    def update_parameters(self, spectrums, stats):
        self.check_inputs(spectrums)
        self.update_lambda(spectrums, stats)
        self.update_phi(spectrums)
        self.update_R(spectrums)

class CGMMTrainer(object):
    def __init__(self, num_bins, time_steps, num_channels):
        self.noise_part = CGMM(num_bins, time_steps, num_channels)
        self.noisy_part = CGMM(num_bins, time_steps, num_channels)
        
    def train(self, spectrums, epoch=30):
        for e in range(1, epoch + 1):
            stats = self.noisy_part.accu_stats(spectrums) + self.noise_part.accu_stats(spectrums)
            self.noise_part.update_parameters(spectrums, stats)
            self.noisy_part.update_parameters(spectrums, stats)
            print('Likelihood: {:.4}'.format(self.noise_part.log_likelihood(spectrums) + \
                    self.noisy_part.Likelihood(spectrums)))

