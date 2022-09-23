import numpy as np
from random import uniform


class ParametricRandomPointCloud(object):

    def __init__(self, p_max, p_min):
        self.max = p_max
        self.min = p_min

    def generatePoint(self):
        x_coordinate = np.random.uniform(self.min[0], self.max[0])
        y_coordinate = np.random.uniform(self.min[1], self.max[1])
        z_coordinate = np.random.uniform(self.min[2], self.max[2])

        return np.array([x_coordinate, y_coordinate, z_coordinate], dtype=np.float64)
