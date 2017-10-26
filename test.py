#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.24

import numpy as np
import utils
import cgmm
from utils import WaveWrapper
from utils import MultiChannelWrapper
from cgmm import CGMMTrainer

def test_multiwrappers():
    wrapper = MultiChannelWrapper('wav.scp')
    print(wrapper)

def test_reconstruction():
    wrapper = WaveWrapper('6ch/F01_050C0103_STR.CH5.wav')
    spectrum = utils.compute_spectrum(wrapper) 
    utils.reconstruct_wave(spectrum, "6ch/F01_050C0103_STR.CH5_rebuild.wav", filter_coeff=0.8)

def test_plot_spectrum():
    wrapper = WaveWrapper('./6ch/F01_050C0103_STR.CH3.wav')
    spectrum = utils.compute_spectrum(wrapper) 
    utils.plot_spectrum(spectrum, wrapper.frame_duration, "F01_050C0103_STR.CH3.wav")
     
def test_train_cgmm():
    wrapper = MultiChannelWrapper('wav.scp')
    print(wrapper)
    (time_steps, num_bins), spectrums = wrapper.spectrums(transpose=True)
    num_bins, time_steps, num_channels = np.array(spectrums).shape
    trainer = CGMMTrainer(num_bins, time_steps, num_channels)
    trainer.train(spectrums, iters=10)

def random_complex_vector(n):
    r = np.random.rand(n)
    i = np.random.rand(n)
    return r + i * 1j

def random_complex_matrix(m, n):
    r = np.random.rand(m, n)
    i = np.random.rand(m, n)
    return np.matrix(r + i * 1j)

def test_complex():
    v = random_complex_vector(4)
    v = np.matrix(v)
    print(v.H * v)

def test_gaussian():
    sigma = random_complex_matrix(6, 6)
    obs   = random_complex_vector(6)
    phi   = 0.1 + 0.5j
    sigma_inv = sigma.I
    sigma_det = np.linalg.det(sigma)
    print cgmm.gmm_posterior(obs, phi, sigma_inv, sigma_det)
    print cgmm.gmm_posterior_slow(obs, phi * sigma)
    
def test_linadet():
    sigma = random_complex_matrix(6, 6)
    phi   = 0.1 + 0.5j
    print np.log(np.linalg.det(phi * sigma))
    print np.log(np.linalg.det(sigma) * (phi ** 6))
    # different from above
    print np.log(np.linalg.det(sigma)) + np.log(phi ** 6)

if __name__ == '__main__':
    # test_complex()
    # test_gaussian()
    # test_linadet()
    test_train_cgmm()
    # test_multiwrappers()
    # test_reconstruction()
    # test_plot_spectrum()
