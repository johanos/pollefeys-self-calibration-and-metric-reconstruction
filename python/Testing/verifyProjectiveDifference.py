# The point of this file is to
# 1. Create a Projective Reconstruction of a series of images
# 2. Auto Calibrate that Projective Reconstruction using the Dual Image of the Absolute Quadric.

# START IMPORTS #
from python.AutoCalibration.projectiveReconstruct import *
import numpy as np
import  random

# END IMPORTS #

def projectPoint(P, xi):
    projection = P @ xi
    projection /= projection[2]
    return projection

def projectPointsWithCam(P, X):
    points = np.ones((X.shape[0], 3))
    for i, xi in enumerate(X):
        points[i,:] = projectPoint(P, xi)
    return points

def projectPoints(Ps, X):
    points2D = np.ones((Ps.shape[0], X.shape[0], 3))
    for i, P in enumerate(Ps):
        points2D[i, :] = projectPointsWithCam(P, X)
    return points2D

def main():
    # need to generate a set of N 3D points

    height = 1000
    width = 1000

    K1 = np.array([[ width,   0, 479.5],
                   [   0, height, 269.5],
                   [   0,   0,     1]])

    K2 = np.array([[width,      0, 479.5],
                   [    0, height, 269.5],
                   [    0,      0,     1]])

    K3 = np.array([[width,      0, 479.5],
                   [    0, height, 269.5],
                   [    0,      0,    1]])

    # 0 degree rotation..
    R1 = np.array( [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])
    # 20 degree rotation
    R2 = np.array( [[ 0.9396926,  0.0000000,  0.3420202],
                    [ 0.0000000,  1.0000000,  0.0000000],
                    [-0.3420202,  0.0000000,  0.9396926]])
    # -20 degree rotation
    R3 = np.array( [[0.9396926, 0.0000000, -0.3420202],
                    [0.0000000, 1.0000000,  0.0000000],
                    [0.3420202, 0.0000000,  0.9396926]])

    t1 = np.array([[0],
                   [0],
                   [0]])

    # in meters
    t2 = np.array([[0.8],
                   [0.5],
                   [0.5]])

    t3 = np.array([[-0.8],
                   [ -0.5],
                   [ 0.0]])

    M1 = np.hstack((R1, t1))
    M2 = np.hstack((R2, t2))
    M3 = np.hstack((R3, t3))

    Ps = np.zeros((3,3,4))

    P1 = K1 @ M1
    Ps[0, :, :] = P1
    P2 = K2 @ M2
    Ps[1, :, :] = P2
    P3 = K3 @ M3
    Ps[2, :, :] = P3

    print(f"\nCAMERA 1:")
    print(f"\t{P1[0, :]}")
    print(f"\t{P1[1, :]}")
    print(f"\t{P1[2, :]}")

    print(f"\nCAMERA 2:")
    print(f"\t{P2[0, :]}")
    print(f"\t{P2[1, :]}")
    print(f"\t{P2[2, :]}")

    print(f"\nCAMERA 3:")
    print(f"\t{P3[0, :]}")
    print(f"\t{P3[1, :]}")
    print(f"\t{P3[2, :]}")

    # I have my cameras... in Ps
    points3D = np.ones((20, 4))
    for pi in range(points3D.shape[0]):
        # generate a random X Y Z
        # my cameras are around 0,0 so points should be somewhere nearby..
        x = random.uniform(-10.5, 10.0)
        y = random.uniform(-10.5, 10.0)
        z = random.uniform(60.5, 80.0)
        points3D[pi,:3] = [x,y,z]


    # Now I need to generate the projected points...


    points2D = projectPoints(Ps, points3D)


    # I now have the projected points

    # need to now compute the fundamental matrix... These point are nice because I have
    # all of them...

    visibilityMat = np.full((points2D.shape[0], points2D.shape[1]), True)

    points1 = points2D[0,:]
    points2 = points2D[1,:]
    points3 = points2D[2,:]

    F12 = computeFundamentalMatrixBetween(points1, points2)
    Pprime1, Pprime2 = createProjectivelyAmbiguousCameraMatrices(points1, points2, F12)

    F13 = computeFundamentalMatrixBetween(points1, points3)
    _, Pprime3 = createProjectivelyAmbiguousCameraMatrices(points1, points3, F13)

    ##

    reconstructed3D =  triangulatePoints(P1, P2, [points1, points2], 0, 1, visibilityMat)
    assert(np.abs(np.sum(np.subtract(points3D, reconstructed3D))) < 0.001)

    reconstructed3D = triangulatePoints(P1, P3, [points1, points3], 0, 2, visibilityMat)
    assert (np.abs(np.sum(np.subtract(points3D, reconstructed3D))) < 0.001)

    error1 = simpleReprojectionError(points3D, np.array([P1, P2]), points2D, [0, 1], visibilityMat)
    print(f"Average Composite Reprojection error P1 P2: {error1}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

    error1 = simpleReprojectionError(points3D, np.array([P1, P3]), points2D, [0, 2], visibilityMat)
    print(f"Average Composite Reprojection error P1 P3: {error1}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")


    projective3D =  triangulatePoints(Pprime1, Pprime2, [points1, points2], 0, 1, visibilityMat)
    error2 = simpleReprojectionError(projective3D, np.array([Pprime1, Pprime2]), points2D, [0, 1], visibilityMat)
    print(f"Average Composite Reprojection error PP1 PP2: {error2}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

    error2 = simpleReprojectionError(projective3D, np.array([Pprime1, Pprime3]), points2D, [0, 2], visibilityMat)
    print(f"Average Composite Reprojection error PP1 PP3 with prev points: {error2}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

    projective3D =  triangulatePoints(Pprime1, Pprime3, [points1, points3], 0, 2, visibilityMat)

    error2 = simpleReprojectionError(projective3D, np.array([Pprime1, Pprime2]), points2D, [0, 1], visibilityMat)
    print(f"Average Composite Reprojection error PP1 PP2: {error2}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

    error2 = simpleReprojectionError(projective3D, np.array([Pprime1, Pprime3]), points2D, [0, 2], visibilityMat)
    print(f"Average Composite Reprojection error PP1 PP3 with prev points: {error2}")
    print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

    # what does this tell me...? that the triangulated points from
    # Pp1, Pp2 choose one set of 3D points
    # Pp1, Pp3 chooses another set of 3D points
    # these do not agree and therefore you can't use them

    # also...

    # P1' * H = P1
    # so H = P1'-1 * P1
    # I can compute the absolute dual

    rectifyingTransform = np.linalg.pinv(Pprime2) @ P2

    print (f"Rectifying transform is:\n{rectifyingTransform}")

    print (f"Original P1:\n{P1}")
    print (f"Fixed one:\n{Pprime1 @ rectifyingTransform}")

    print(f"Original P2:\n{P2}")
    print(f"Fixed one:\n{Pprime2 @ rectifyingTransform}")

    print(f"Original P3:\n{P3}")
    print(f"Fixed one:\n{Pprime3 @ rectifyingTransform}")




if __name__ == '__main__':
    main()