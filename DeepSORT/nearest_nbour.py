#!/usr/bin/env python3

import numpy as np

def cosineDist(a, b, normalized=False):

    if not normalized:
        a = np.asarray(a)/np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(a)/np.linalg.norm(b, axis=1, keepdims=True)

    return 1. - np.dot(a, b.T)

def nnCosineDist(x, y):

    d = cosineDist(x, y)

    return d.min(axis=0)

def euclideanDist(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    a2 = np.square(a).sum(axis=1)
    b2 = np.square(b).sum(axis=1)

    d = -2. * np.dot(a, b.T) + a2[:,None] + b2[None,:]
    d = np.clip(d, 0., float(np.inf))

    return d

def nnEuclideanDist(x, y):

    d = euclideanDist(x, y)

    return np.maximum(0.0, d.min(axis=0))


class NNDistance():
    """
    A nearest neighbor distance metric that, for each target, returns the
    closest distance to any sample that has been observed so far

    """

    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "euclidean":
            self.metric_ = nnEuclideanDist
        elif metric == "cosine":
            self.metric_ = nnCosineDist
        else:
            raise ValueError("Invalid metric: should be 'euclidean' or 'cosine'")

        self.matching_threshold_ = matching_threshold_
        self.budget_ = budget
        self.samples_ = {}

    def partialFit(self, features, targets, active_targets):

        for feature, target in zip(features, targets):
            self.samples_.setdefault(target, []).append(feature)

            if self.budget_ is not None:
                self.samples_[target] = self.samples_[target][-self.budget_:]

        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):

        cost = np.zeros(len(targets), len(features))

        for i, target in enumerate(targets):
            cost[i,:] = self.metric_(self.samples[target], features)

        return cost
