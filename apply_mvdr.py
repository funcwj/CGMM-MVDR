#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.27

import argparse
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
    sigma_noisy = np.load(args.sigma_noisy)
    sigma_noise = np.load(args.sigma_noise)
    sigma_clean = sigma_noisy - sigma_noise

    wrapper = MultiChannelWrapper(args.descriptor)
    (time_steps, num_bins), spectrums = wrapper.spectrums()
    specs_noisy = np.transpose(spectrums, (2, 1, 0)) 
    specs_enhan = np.zeros([num_bins, time_steps]).astype(np.complex)
    for f in range(num_bins):
        steer_vector = beamformer.main_egvec(sigma_clean[f])
        specs_enhan[f] = beamformer.apply_mvdr(steer_vector, sigma_noise[f], specs_noisy[f]) 
    utils.reconstruct_wave(np.transpose(specs_enhan), args.save_dir, filter_coeff=args.filter_coeff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply CGMM-MVDR beamformer on multiple channel")
    parser.add_argument('descriptor', type=str,
                        help="""descriptor of multiple channel location""")
    parser.add_argument('sigma_noisy', type=str,
                        help="""sigma of noisy(noise + clean) part estimated by CGMM""")
    parser.add_argument('sigma_noise', type=str,
                        help="""sigma of noise part estimated by CGMM""")
    parser.add_argument('-s', '--save',
                        dest='save_dir', type=str, default='default.wav',
                        help="""path to save the enhanced wave""")
    parser.add_argument('-c', '--filter_coeff',
                        dest='filter_coeff', type=float, default='0.97',
                        help="""filter coefficient to apply when reconstruct wave""")
    args = parser.parse_args()
    main(args)

