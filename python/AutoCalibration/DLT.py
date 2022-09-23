import numpy as np
from scipy import linalg
from typing import List


# TODO: make this change, verify it works on calibrationPipeline and delete the old one I will cleanse this codebase of this linguini code now that i'm more comfortable
def isotropic_2D_scaling_transform(points_2d: np.ndarray) -> np.ndarray:
    """
    :param points_2d: np array of 2d points to get a zero mean and std of sqrt(2)
    :return: transform to give zero mean and sqrt(2) std (ndarray)
    """

    N = points_2d.shape[0]
    # make sure we don't take the homogeneous coordinate
    points_uv = points_2d[:, :2]
    mean = np.mean(points_uv, axis=0)
    centered_points = points_uv - mean
    rms = np.sum(centered_points**2) / N
    scale = np.sqrt(2.0 / rms * 1.0)

    #  np.array([[scale,     0, -mean[0] * scale],
    #            [    0, scale, -mean[1] * scale],
    #            [    0,     0,                1]])

    scaling_transformation = np.array([[scale, 0, -mean[0] * scale], [0, scale, -mean[1] * scale], [0, 0, 1]])
    return scaling_transformation


# TODO: make this change, verify it works on calibrationPipeline and delete the old one I will cleanse this codebase of this linguini code now that i'm more comfortable
def isotropic_3D_scaling_transform(points_3d) -> np.ndarray:
    """
    :param points_3d: np array of 3d points to get a zero mean and std of sqrt(2)
    :return: transform to give zero mean and sqrt(2) std (ndarray)
    """

    N = points_3d.shape[0]
    points_xyz = points_3d[:, :3]
    mean = np.mean(points_xyz, axis=0)
    centeredPoints = points_xyz - mean
    rms = np.sum(centeredPoints**2) / N * 1.0
    scale = np.sqrt(3 / rms * 1.0)

    # np.array([[scale,     0,     0, -mean[0] * scale],
    #           [    0, scale,     0, -mean[1] * scale],
    #           [    0,     0, scale, -mean[2] * scale],
    #           [    0,     0,     0,                1]])

    scaling_transformation = np.array([[scale, 0, 0, -mean[0] * scale], [0, scale, 0, -mean[1] * scale],
                                       [0, 0, scale, -mean[2] * scale], [0, 0, 0, 1]])
    return scaling_transformation


# from H.Z pg 107 centralize image points to 0 and have std dev of sqrt(2)
def isotropic_2D_scaling_transformation_for(points2D, view, visibilityMatrix):
    pointsUV = None
    for i, point in enumerate(points2D):
        if visibilityMatrix[view, i]:
            if pointsUV is not None:
                pointsUV = np.vstack((pointsUV, point[:2]))
            else:
                pointsUV = point[:2]

    N = pointsUV.shape[0]
    mean = np.mean(pointsUV, axis=0)
    centeredPoints = pointsUV - mean
    rms = np.sum(centeredPoints**2) / N
    scale = np.sqrt(2 / rms * 1.0)

    IST = np.array([[scale, 0, -mean[0] * scale], [0, scale, -mean[1] * scale], [0, 0, 1]])
    return IST


def isotropic3DScalingTransformationFor(points3d, view, visibilityMatrix):
    pointsXYZ = None
    for i, point in enumerate(points3d):
        if visibilityMatrix[view, i]:
            if pointsXYZ is not None:
                pointsXYZ = np.vstack((pointsXYZ, point[:3]))
            else:
                pointsXYZ = point[:3]

    N = pointsXYZ.shape[0]
    pointsXYZ = points3d[:, :3]
    mean = np.mean(pointsXYZ, axis=0)
    centeredPoints = pointsXYZ - mean
    rms = np.sum(centeredPoints**2) / N * 1.0
    scale = np.sqrt(3 / rms * 1.0)
    IST = np.array([[scale, 0, 0, -mean[0] * scale], [0, scale, 0, -mean[1] * scale],
                    [0, 0, scale, -mean[2] * scale], [0, 0, 0, 1]])
    return IST


def DLT3D_Resection(points_3D, points_2D):
    """
    Get a camera matrix that aligns with the observed 3D points and their 2D projections based on page 181 in H.Z book
    :param points_3D: homogeneous coordinates in the object 3D space.
    :param points_2D: homogeneous coordinates in the image 2D space.
    :return: camera matrix ndarray (3,4)
    """

    N = points_3D.shape[0]

    if N < 6:
        raise ValueError(f"3D DLT requires at least 6 3D points. Only {N} were given")

    # such that xi_norm = T@xi
    similarity_transformation_2d = isotropic_2D_scaling_transform(points_2D)
    norm_points_2d = (similarity_transformation_2d @ points_2D.T).T

    # such that Xi_norm = U@Xi
    similarity_transformation_3d = isotropic_3D_scaling_transform(points_3D)
    norm_points_3d = (similarity_transformation_3d @ points_3D.T).T

    A = []

    for i in range(N):
        x, y, z, rho = norm_points_3d[i, 0], norm_points_3d[i, 1], norm_points_3d[i, 2], norm_points_3d[i, 3]
        u, v, w = norm_points_2d[i, 0], norm_points_2d[i, 1], norm_points_2d[i, 2]

        A.append([0, 0, 0, 0, (-w * x), (-w * y), (-w * z), (-w * 1), (v * x), (v * y), (v * z), (v * rho)])
        A.append([(w * x), (w * y), (w * z), (w * 1), 0, 0, 0, 0, (-u * x), (-u * y), (-u * z), (-u * rho)])

    # Convert A to np arrayfiltered_3d
    A = np.asarray(A)

    # Find the parameters of P:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    # # Camera projection matrix
    Hbar = L.reshape(3, 4)
    #
    # Denormalization
    # maybe use pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = (np.linalg.inv(similarity_transformation_2d) @ Hbar) @ similarity_transformation_3d

    P = H / H[-1, 2]

    return P


# TODO: Create a better way for the points to be cleaned up rather than passing the Vmat everywhere...
def DLTcalib(points3D, points2D, view, visibilityMatrix):
    '''
    Camera calibration by DLT using known object points and their image points.
    Input
    -----
    points3D: homogeneous coordinates in the object 3D space.
    points2D: homogeneous coordinates in the image 2D space.
    The coordinates (<x,y,z,1> and <u,v,1>) are given as columns and the different points as rows.
    There must be at least 6 calibration points for the 3D DLT.
    Output
    ------
     L: array of 11 parameters of the calibration matrix.
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''

    # Converting all variables to numpy array

    n = points3D.shape[0]

    if n < 6:
        raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %
                         (3, 6, n))

    # s.t xiBar = Txi
    T = isotropic_2D_scaling_transformation_for(points2D, view, visibilityMatrix)
    tPoints2D = (T @ points2D.T).T
    # s.t Xibar = UXi
    U3D = isotropic3DScalingTransformationFor(points3D, view, visibilityMatrix)
    tPoints3D = (U3D @ points3D.T).T

    A = []
    for i in range(n):
        x, y, z, rho = tPoints3D[i, 0], tPoints3D[i, 1], tPoints3D[i, 2], tPoints3D[i, 3]
        u, v, w = tPoints2D[i, 0], tPoints2D[i, 1], tPoints2D[i, 2]

        A.append([0, 0, 0, 0, (-w * x), (-w * y), (-w * z), (-w * 1), (v * x), (v * y), (v * z), (v * rho)])
        A.append([(w * x), (w * y), (w * z), (w * 1), 0, 0, 0, 0, (-u * x), (-u * y), (-u * z), (-u * rho)])

    # Convert A to np array
    A = np.asarray(A)

    # Find the parameters of P:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]
    # # Camera projection matrix
    Hbar = L.reshape(3, 4)
    #
    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = (np.linalg.inv(T) @ Hbar) @ U3D

    P = H / H[-1, 2]
    return P


def factor(P):
    """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
    # factor first 3*3 part
    K, R = linalg.rq(P[:, :3])
    K = K / K[-1, -1]
    # make diagonal of K positive
    T = np.diag(np.sign(np.diag(K)))
    if linalg.det(T) < 0:
        T[1, 1] *= -1

    K = K @ T
    R = T @ R  # T is its own inverse
    t = linalg.pinv(K) @ P[:, 3]

    for j in range(K.shape[0]):
        if K[j, j] < 0:
            for i in range(K.shape[1]):
                K[i, j] = -K[i, j]

    return K, R, t.reshape(3, 1)
