# This file is going to test how well this auto calibration works for dome configurations with a certain shape at the origin....
import numpy as np
import math
from math import pi as PI
from python.ParametricObjects.ParametricDome import ParametricDome as Dome
from python.ParametricObjects.ParametricPlane import ParametricPlane
from python.ParametricObjects.ParametricCube import ParametricCube
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib as mpl
import  matplotlib.pyplot as plt
mpl.use('macosx')

def main():
    dome = Dome(2,2,2)

    inputsU = np.linspace(0, 2 * PI, 10, dtype=np.float)
    inputsV = np.linspace(0, PI / 2, 5, dtype=np.float)
    inputsUU, inputsVV = np.meshgrid(inputsU, inputsV)
    positions = []
    for i in range(len(inputsU)):
        for j in range(len(inputsV)):
            u = inputsUU[j, i]
            v = inputsVV[j, i]
            domePos = dome.generatePoint(u,v)
            positions.append(domePos)

    positions = np.array(positions)

    minCube = [-0.5, -0.5, 0]
    maxCube = [0.5, 0.5, 1]

    cube = ParametricCube(minCube, maxCube)

    planeU = np.linspace(0, 1, 4, dtype=np.float)
    planeV = np.linspace(0, 1, 4, dtype=np.float)
    planeUU, planeVV = np.meshgrid(planeU, planeV)
    planePos = []
    for w in range(6):
        for i in range(len(planeU)):
            for j in range(len(planeV)):
                u = planeUU[j, i]
                v = planeVV[j, i]

                planeP = cube.generatePoint(u, v, w)
                planePos.append(planeP)


    planePos = np.array(planePos)



    figure = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.set_aspect('equal')
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=positions[:,2], cmap="hsv")
    ax.scatter(planePos[:, 0], planePos[:, 1], planePos[:, 2], cmap="hsv")
    plt.ion()
    plt.show()


if __name__ == '__main__':
    main()






















