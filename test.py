#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.24

import numpy as np
import utils
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
    trainer.train(spectrums, epoch=10)


if __name__ == '__main__':
    test_train_cgmm()
    # test_multiwrappers()
    # test_reconstruction()
    # test_plot_spectrum()
