#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.26

import argparse
import time
import numpy as np

from utils import MultiChannelWrapper
from cgmm import CGMMTrainer

def train(args):
    wrapper = MultiChannelWrapper(args.descriptor)  
    (time_steps, num_bins), spectrums = wrapper.spectrums()
    trainer = CGMMTrainer(num_bins, time_steps, len(spectrums))
    start_time = time.time()
    trainer.train(np.transpose(spectrums), iters=args.iters)
    finish_time = time.time()
    print('Total raining time: {:.3f}s'.format(finish_time - start_time))
    trainer.save_param(args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training CGMM on multiple channel")
    parser.add_argument('descriptor', type=str,
                        help="""descriptor of multiple channel location""")
    parser.add_argument('-i', '--iters',
                        dest='iters', type=int, default='10',
                        help="""number of iterations to train""")
    parser.add_argument('-s', '--save',
                        dest='save_dir', type=str, default='',
                        help="""directory to save sigma of CGMM""")
    args = parser.parse_args()
    train(args)
