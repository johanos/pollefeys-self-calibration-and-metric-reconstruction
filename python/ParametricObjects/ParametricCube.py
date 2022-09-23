from python.ParametricObjects.ParametricPlane import ParametricPlane
import numpy as np

class ParametricCube(object):
    def __init__(self, min, max):
        self.faces = []

        bottom_plane = ParametricPlane([min[0], min[1], min[2]],
                                       [min[0], max[1], min[2]],
                                       [max[0], min[1], min[2]],
                                       [max[0], max[1], min[2]]  )

        top_plane = ParametricPlane(   [min[0], min[1], max[2]],
                                       [min[0], max[1], max[2]],
                                       [max[0], min[1], max[2]],
                                       [max[0], max[1], max[2]]  )

        front_face = ParametricPlane(  [min[0], min[1], min[2]],
                                       [max[0], min[1], min[2]],
                                       [min[0], min[1], max[2]],
                                       [max[0], min[1], max[2]]  )

        back_face = ParametricPlane( [min[0], max[1], min[2]],
                                     [max[0], max[1], min[2]],
                                     [min[0], max[1], max[2]],
                                     [max[0], max[1], max[2]]   )

        left_face = ParametricPlane([min[0], min[1], min[2]],
                                    [min[0], max[1], min[2]],
                                    [min[0], min[1], max[2]],
                                    [min[0], max[1], max[2]])

        right_face = ParametricPlane( [max[0], min[1], min[2]],
                                      [max[0], max[1], min[2]],
                                      [max[0], min[1], max[2]],
                                      [max[0], max[1], max[2]]  )

        self.faces.append(top_plane)
        self.faces.append(bottom_plane)
        self.faces.append(front_face)
        self.faces.append(back_face)
        self.faces.append(left_face)
        self.faces.append(right_face)



    def generatePoint(self, u, v, w):
        point = self.faces[w].generatePoint(u,v)
        x_coordinate = point[0]
        y_coordinate = point[1]
        z_coordinate = point[2]

        return np.array([x_coordinate, y_coordinate, z_coordinate], dtype=np.float64)