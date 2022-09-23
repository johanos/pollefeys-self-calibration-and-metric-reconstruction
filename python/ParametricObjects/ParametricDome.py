import math
import numpy as np

class ParametricDome(object):

    def __init__(self, rx, ry, rz):
        self.rx = rx
        self.ry = ry
        self.rz = rz


    def generatePoint(self, u , v ):

        x_coordinate = self.rx * math.cos(v) * math.cos(u)
        y_coordinate = self.ry * math.cos(v) * math.sin(u)
        z_coordinate = self.rz * math.sin(v)

        return np.array([x_coordinate, y_coordinate, z_coordinate], dtype=np.float64)



