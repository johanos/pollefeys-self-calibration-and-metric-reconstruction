import numpy as np
class ParametricPlane(object):
    def __init__(self, p_0, p_1, p_2, p_3):
        self.p_0 = p_0
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3


    def generatePoint(self, u, v):
        x_coordinate = (1 - v) * ((1 - u) * self.p_0[0] + u * self.p_1[0]) + v * ((1 - u) * self.p_2[0] + u * self.p_3[0])
        y_coordinate = (1 - v) * ((1 - u) * self.p_0[1] + u * self.p_1[1]) + v * ((1 - u) * self.p_2[1] + u * self.p_3[1])
        z_coordinate = (1 - v) * ((1 - u) * self.p_0[2] + u * self.p_1[2]) + v * ((1 - u) * self.p_2[2] + u * self.p_3[2])

        return np.array([x_coordinate, y_coordinate, z_coordinate], dtype=np.float64)