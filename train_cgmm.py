#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.26

import argparse
import numpy as np

from utils import MultiChannelWrapper
from cgmm import CGMMTrainer

def train(args):
    wrapper = MultiChannelWrapper(args.descriptor)  
    (time_steps, num_bins), spectrums = wrapper.spectrums(transpose=True)
    num_bins, time_steps, num_channels = np.array(spectrums).shape
    trainer = CGMMTrainer(num_bins, time_steps, num_channels)
    trainer.train(spectrums, iters=args.iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training CGMM on multiple channel")
    parser.add_argument('descriptor', type=str,
                        help="""descriptor of multiple channel location, format:
                                /path/to/channel1
                                ...
                                /path/to/channeln""")
    parser.add_argument('-i', '--iters',
                        dest='iters', type=int, default='10',
                        help="""number of iterations to train""")
    args = parser.parse_args()
    train(args)
