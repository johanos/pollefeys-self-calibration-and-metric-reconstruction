# The point of this file is to
# 1. Create a Projective Reconstruction of a series of images
# 2. Auto Calibrate that Projective Reconstruction using the Dual Image of the Absolute Quadric.

# START IMPORTS #
from python.AutoCalibration.calibrationAPI import self_calibrate
from python.AutoCalibration.ProjectiveReconstructionHelpers import *
import numpy as np
from python.AutoCalibration.DLT import DLTcalib
from scipy.stats import special_ortho_group
from random import uniform
from python.AutoCalibration.Camera import Camera
from python.AutoCalibration.calibrationAPI import *

# END IMPORTS #

HProjective = np.array([[1, 0, -4, 2], [1, -1, 1, -7], [-4, 0, 2, 0], [1, 2, 3, 1]], dtype=np.float32)


def calculate_reprojection_error(points_3d_collection, cameras, points_uv_collection, visibility_matrix):
    """
    :param points_3d_collection: 3D points to look at
    :param cameras: collection of camera objects for the cameras I am using
    :param points_uv_collection: collection of uv_points in homogeneous coordinates
    :param visibility_matrix:  a visibility matrix of size num_views x num_3d_points
    :return: reprojection_error - sum of vis[i,j] * (projection(cam[j], 3d_point[i], uv_measurement[i,j])
    """

    total_error = 0.0
    total_entries = 0
    cam: Camera  # for type hints
    for i, cam in enumerate(cameras):
        points_uv_for_view = points_uv_collection[i, :]

        for j, point_3d in enumerate(points_3d_collection):
            is_visible = visibility_matrix[i, j]
            if not is_visible:
                continue

            projection = cam.project(point_3d.reshape(4, 1), False, False).reshape(3)
            # calculate the distance between projection and the point uv
            point_uv = points_uv_for_view[j]

            squared_distance = np.linalg.norm(projection - point_uv)**2
            total_error += squared_distance
            total_entries += 1

    return total_error, total_entries


def projectPoint(P, xi):
    projection = P @ xi
    projection /= projection[2]
    return projection


def projectPointsWithCam(P, X):
    points = np.ones((X.shape[0], 3))
    for i, xi in enumerate(X):
        points[i, :] = projectPoint(P, xi)
    return points


def projectPoints(Ps, X):
    points2D = np.ones((Ps.shape[0], X.shape[0], 3))
    for i, P in enumerate(Ps):
        points2D[i, :] = projectPointsWithCam(P, X)
    return points2D


def main():
    # need to generate a set of N 3D points
    numViews = 4

    numPoints = 30

    global Ks

    Ks = np.zeros((numViews, 3, 3))
    Rs = np.zeros((numViews, 3, 3))
    Ts = np.zeros((numViews, 3, 1))

    Ps = np.zeros((numViews, 3, 4))

    cameras = []
    width = 0
    height = 0
    for i in range(numViews):
        width = 960  #uniform(300,1000)
        height = 540  #uniform(300,1000)
        focal = uniform(900, 1000)
        focal2 = focal

        K = np.array([[focal, 0, (width - 1) / 2], [0, focal2, (height - 1) / 2], [0, 0, 1]],
                     dtype=np.float32)

        print(f"K_{i}:\n{K}")
        Ks[i, :, :] = K  # np.copy(flyingThingsIntrinsicMatrix)
        #print(Ks[i,:,:])

        Rs[i, :, :] = special_ortho_group.rvs(3)
        R = Rs[i, :, :]

        Ts[i, :, :] = np.array([[uniform(-10, 10)], [uniform(-10, 10)], [uniform(-8, 8)]], dtype=np.float64)
        t = Ts[i, :, :]

        M = np.hstack((Rs[i], Ts[i]))
        Ps[i, :, :] = Ks[i] @ M
        cam = Camera(i, K, R, t, None)

        cameras.append(cam)

    for i in range(numViews):

        print(f"\nCAMERA {i}:")
        print(f"\t{Ps[i, 0, :]}")
        print(f"\t{Ps[i, 1, :]}")
        print(f"\t{Ps[i, 2, :]}")

    # I have my cameras... in Ps

    points3D = np.ones((numPoints, 4), dtype=np.float64)
    for pi in range(points3D.shape[0]):
        # generate a random X Y Z
        # my cameras are around 0,0 so points should be somewhere nearby..
        x = uniform(-10.5, 10.0)
        y = uniform(-10.5, 10.0)
        z = uniform(1.0, 4.0)
        points3D[pi, :3] = [x, y, z]

    # Now I need to generate the projected points...
    points2D = projectPoints(Ps, points3D)
    points_2 = []

    for cam in cameras:
        points_uv = cam.project(points3D.T, False, False).T
        points_2.append(points_uv)

    points_2 = np.array(points_2)
    assert (np.all(points_2 - points2D < 1e-4))

    # I now have the projected points
    # need to now compute the fundamental matrix... These point are nice because I have
    # all of them...
    visibilityMat = np.full((points2D.shape[0], points2D.shape[1]), True)
    # visibilityMat[1, 2:4] = False
    # visibilityMat[2, 6] = False
    # visibilityMat[1, 7] = False

    points_1_2_mask = np.logical_and(visibilityMat[0], visibilityMat[1]).astype(np.uint8)

    points0 = points2D[0, :][points_1_2_mask.ravel() == 1]
    points1 = points2D[1, :][points_1_2_mask.ravel() == 1]

    reconstructed3D = triangulate_points_no_visibility_matrix(Ps[0], Ps[1], [points0, points1])
    p_3d_gt = points3D[points_1_2_mask.ravel() == 1]
    assert (np.abs(np.sum(np.subtract(p_3d_gt, reconstructed3D))) < 0.001)

    for i in range(1, numViews):
        viz_mask = np.logical_and(visibilityMat[0], visibilityMat[i]).astype(np.uint8)
        points_i_2d = points2D[i, :][viz_mask.ravel() == 1]
        points0 = points2D[0, :][viz_mask.ravel() == 1]
        reconstructed3D = triangulate_points_no_visibility_matrix(Ps[0, :, :], Ps[i, :, :],
                                                                  [points0, points_i_2d])

        p_3d_gt = points3D[viz_mask.ravel() == 1]
        assert (np.abs(np.sum(np.subtract(p_3d_gt, reconstructed3D))) < 0.001)

    # Can I resection a P using DLT to have the same projection matrix P3...

    for i in range(numViews):
        #Pnew = DLTcalib(points3D, points2D[i, :], i, visibilityMat)

        Pnew = DLT3D_Resection(points3D, points2D[i, :])
        viz_mask = np.logical_and(visibilityMat[0], visibilityMat[2])
        Pnew_2 = DLT3D_Resection(points3D[viz_mask.ravel() == 1], points2D[i][viz_mask.ravel() == 1])
        cam = Camera(i, None, None, None, Pnew)
        with np.printoptions(precision=4, suppress=True):
            print(f"Old P{i}:\n{Ps[i, :, :] / Ps[i, -1, -1]}\nNew P{i}:\n{Pnew / Pnew[-1,-1] }")
            print(f"P{i}_2:\n{Pnew_2/ Pnew_2[-1, -1]}")
            error1, total = calculate_reprojection_error(points3D, [cam],
                                                         points2D[i].reshape(1, points2D[i].shape[0], 3),
                                                         visibilityMat[i].reshape(1, -1))
            print(f"\tReprojection error on new P{i}: {error1:4f} average {(error1/total):8f}")

        assert (np.abs(np.sum(np.subtract(Ps[i, :, :] / Ps[i, -1, -1], Pnew / Pnew[-1, -1]))) < 0.01)

    print(f"TESTING IF A WE CAN RECOVER CAM CALIB CORRECTLY FOR UNIFORM PROJECTIVE TRANSFORM")
    # Test Pollefeys method works
    projMats = np.zeros((numViews, 3, 4))
    for i, cam in enumerate(cameras):
        P_dist = cam.P @ np.linalg.inv(HProjective)
        projMats[i, :, :] = P_dist

    # distorted ProjMats..s

    Ks, isDegen, maxD = self_calibrate(projMats, width, height, Ks[0])

    for K in Ks:

        print(f"Ks:\n{K}")

    print(f"==" * 20 + " END " + "==" * 20)
    points0 = points2D[0][points_1_2_mask.ravel() == 1]
    points1 = points2D[1][points_1_2_mask.ravel() == 1]
    F01, mask = compute_fundamental_matrix_between(points0, points1)
    P_prime_0, P_prime_1 = create_projectively_ambiguous_camera_matrices(F01)
    cameras = []
    cam_0: Camera = Camera(0, None, None, None, P_prime_0)
    cam_1: Camera = Camera(1, None, None, None, P_prime_1)

    cameras.append(cam_0)
    cameras.append(cam_1)
    combined_points = np.array([points0, points1])

    projective_3d = triangulate_points_no_visibility_matrix(P_prime_0, P_prime_1, combined_points)

    error, num_items = calculate_reprojection_error(projective_3d, cameras, np.array([points0, points1]),
                                                    visibilityMat)
    print(f"Reprojection error PP1 PP2: {error} rounded to 7 decimals: {error:7f}")
    print(
        f"Average Reprojection error PP1 PP2: {error / num_items} rounded to 7 decimals: {error / num_items:7f}"
    )

    for i in range(2, numViews):
        candidate_matches_mask = np.logical_and(points_1_2_mask, visibilityMat[i])
        filtered_3d = projective_3d[candidate_matches_mask.ravel() == 1]
        filtered_2d = points2D[i][candidate_matches_mask.ravel() == 1]
        P = DLT3D_Resection(filtered_3d, filtered_2d)
        cam: Camera = Camera(i, None, None, None, P)
        cameras.append(cam)

    error, num_items = calculate_reprojection_error(projective_3d, cameras, points2D, visibilityMat)

    print(f"Reprojection error PP1 PP2: {error} rounded to 7 decimals: {error:7f}")
    print(
        f"Average Reprojection error PP1 PP2: {error/num_items} rounded to 7 decimals: {error/num_items:7f}")

    Ps = []

    for cam in cameras:
        Ps.append(cam.P)

    Ps = np.array(Ps)
    Ks, is_degen, max_dist = self_calibrate(Ps, width, height, cameras[0].K)

    for K in Ks:

        print(f"Ks:\n{K}")


if __name__ == '__main__':
    main()
