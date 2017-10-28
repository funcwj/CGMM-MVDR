#!/usr/bin/env python
# coding=utf-8
# wujian@17.10.26

import numpy as np

def main_egvec(mat):
    """
        return the eigen vector which has maximum eigen value
    """
    assert mat.ndim == 2, "Input must be 2-dim matrix/ndarray"
    eigen_val, eigen_vec = np.linalg.eig(mat)
    max_index = np.argsort(eigen_val.real)[-1]
    return eigen_vec[max_index]

def apply_mvdr(steer_vector, sigma_noise, spectrum_onbin):
    """
        inputs:
            steer_vector:   M x 1 => d
            sigma_noise:    M x M => \phi_v
            spectrum_onbin: T x M => y
        w = \phi_v^{-1} * d / (d^H * \phi_v^{-1} * d) => M x 1
        s = w^H * y^T => 1 x T
    """
    # T x M => M x T
    y = np.matrix(spectrum_onbin).T
    # 1 x M => M x 1
    d = np.matrix(steer_vector).T
    phi_inv = np.matrix(sigma_noise).I
    # M x 1
    w = phi_inv * d / (d.H * phi_inv * d)
    s = w.H * y
    return s

    
    
