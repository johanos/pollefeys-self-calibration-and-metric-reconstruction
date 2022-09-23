from scipy import linalg
import numpy as np


class Camera(object):

    def __init__(self, id, K, R, t, P=None):
        self.id = id
        if P is not None:
            self.P = P
            self.K = None
            self.R = None
            self.t = None
        else:
            assert (K.shape == (3, 3))
            self.K = K
            assert (R.shape == (3, 3))
            self.R = R
            assert (t.shape == (3, 1))
            self.t = t
            self.P = K @ np.hstack((R, t))

        assert (self.P.shape == (3, 4))

    def project(self, X, generate_noise, sigma):
        """Project points in X (4*n) array and normalize the coordinates additionally add gaussian noise to thes"""
        x = self.P @ X
        num_samples = X.shape[1]
        for i in range(3):
            x[i] /= x[2]

        if generate_noise:
            # do something with the noise....
            for i in range(2):
                # create noise with numpy
                noise = np.random.normal(0, sigma, num_samples)
                x[i] += noise

        return x
