#!/usr/bin/env python3

import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1,...,9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
                1: 3.8415,
                2: 5.9915,
                3: 7.8147,
                4: 9.4877,
                5: 11.070,
                6: 12.592,
                7: 14.067,
                8: 15.507,
                9: 16.919
            }

class KalmanFilter:
    """
    A Kalman filter class for tracking the bounding box in image space

    8-dimensional state space
        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as a direct observation of the state space
    """

    def __init__(self):

        self.dim_ = 2

        self.A_ = np.eye(2*self.dim_)

        for i in range(self.dim_):
            self.A_[i, self.dim_+i] = 1.

        self.C_ = np.eye(self.dim_, 2*self.dim_)

        self.pos_std_ = 1./20
        self.vel_std_ = 1./160

    def initialize(self, measurement):

        pos_mean = measurement
        vel_mean = np.zeros_like(pos_mean)

        mean = 
