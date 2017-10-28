#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.27

import argparse
import scipy.io as sio
import numpy as np
import beamformer
import utils
from utils import MultiChannelWrapper

def main(args):
    """
        M: num_chanels, T: num_frames
        apply_mvdr inputs:
            steer_vector:     1 x M
            sigma_noise[f]:      M x M
            spectrum_onbin[f]:   T x M
            return 1 x T
    """
    wrapper = MultiChannelWrapper(args.descriptor)
    (time_steps, num_bins), spectrums = wrapper.spectrums()
    num_channels = len(spectrums)
    specs_noisy = np.transpose(spectrums, (2, 1, 0)) 
    noisy_covar = np.zeros([num_bins, num_channels, num_channels]).astype(np.complex)
    noise_covar = np.zeros([num_bins, num_channels, num_channels]).astype(np.complex)
    noise_lambda = np.load(args.noise_lambda)
    for f in range(num_bins):
        sum_lambda = noise_lambda[f].sum()
        for t in range(time_steps):
            y = np.matrix(specs_noisy[f, t])
            noisy_covar[f] += y.H * y
            noise_covar[f] += (y.H * y * noise_lambda[f, t])
        noise_covar[f] /= sum_lambda
        noisy_covar[f] /= time_steps
    clean_covar = noisy_covar - noise_covar 
    specs_enhan = np.zeros([num_bins, time_steps]).astype(np.complex)
    for f in range(num_bins):
        steer_vector = beamformer.main_egvec(clean_covar[f])
        specs_enhan[f] = beamformer.apply_mvdr(steer_vector, noise_covar[f], specs_noisy[f]) 
    sio.savemat('specs_enhan.mat', {'specs': specs_enhan})
    utils.reconstruct_wave(np.transpose(specs_enhan), args.save_dir, filter_coeff=args.filter_coeff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply CGMM-MVDR beamformer on multiple channel")
    parser.add_argument('descriptor', type=str,
                        help="""descriptor of multiple channel location""")
    parser.add_argument('noise_lambda', type=str,
                        help="""lambda of noise part estimated by CGMM""")
    parser.add_argument('-s', '--save',
                        dest='save_dir', type=str, default='ENHAN_BY_PYTHON.wav',
                        help="""path to save the enhanced wave""")
    parser.add_argument('-c', '--filter_coeff',
                        dest='filter_coeff', type=float, default='0.97',
                        help="""filter coefficient to apply when reconstruct wave""")
    args = parser.parse_args()
    main(args)

