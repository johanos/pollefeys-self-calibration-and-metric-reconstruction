# This file is going to test how well this auto calibration works for dome configurations with a certain shape at the origin....
import numpy as np

from python.AutoCalibration.projectiveReconstruct import *
from python.AutoCalibration.Camera import Camera
from python.AutoCalibration.ProjectiveReconstructionHelpers import skew
from python.ParametricObjects.ParametricRandomPointCloud import ParametricRandomPointCloud
import mpl_toolkits.mplot3d as plt3d
from random import uniform

import matplotlib.pyplot as plt
from python.AutoCalibration.flyingThingsUtils import *
from python.AutoCalibration.calibrationAPI import self_calibrate
from scipy.linalg import expm


def plot_cam_axis(cam, three_d_ax):
    R = cam.R.T
    cam_position = -(R @ cam.t).reshape(3)
    three_d_ax.scatter([cam_position[0]], [cam_position[1]], [cam_position[2]],
                       c=[cam_position[2]],
                       cmap="rainbow")

    length = 0.5
    x_line = cam_position + R[:, 0] * length  # first basis vector..
    y_line = cam_position + R[:, 1] * length  # second basis vector..
    z_line = cam_position + R[:, 2] * length * 1.5  # third basis vector..

    x_axis = plt3d.axes3d.art3d.Line3D((x_line[0], cam_position[0]), (x_line[1], cam_position[1]),
                                       (x_line[2], cam_position[2]),
                                       color="r")

    y_axis = plt3d.axes3d.art3d.Line3D((y_line[0], cam_position[0]), (y_line[1], cam_position[1]),
                                       (y_line[2], cam_position[2]),
                                       color="g")

    z_axis = plt3d.axes3d.art3d.Line3D((z_line[0], cam_position[0]), (z_line[1], cam_position[1]),
                                       (z_line[2], cam_position[2]),
                                       color="b")

    three_d_ax.add_line(x_axis)
    three_d_ax.add_line(y_axis)
    three_d_ax.add_line(z_axis)


def normalize_vector(v):
    if np.all(v == 0):
        return v
    return v / np.linalg.norm(v)


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max(
        [abs(lim - mean_) for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean)) for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


STATIC_F = 1000
IM_HEIGHT = 1000
IM_WIDTH = 1000


def run_experiment(K_sequence, generate_noise, sigma, plot):
    min_range = np.array([-3, -10, -2])
    max_range = np.array([3, 10, 12])

    num_points = 20
    object_points = []
    cloud = ParametricRandomPointCloud(min_range, max_range)
    for i in range(num_points):
        cloud_point = cloud.generatePoint()
        object_points.append(cloud_point)

    object_points = np.array(object_points)
    assert (object_points.shape[0] == num_points)
    # now need to project using these extrinsics and points...
    object_points = np.hstack((object_points, np.ones((object_points.shape[0], 1))))

    # principal axis is z forward. principal point will be at 0.5 0.5 and
    # origin after shift should be at bottom right at 0,0

    # just do pure rotation for now...

    num_rotations = K_sequence.shape[0]
    points_2d = np.ones((num_rotations, object_points.shape[0], 3))

    if plot:
        fig = plt.figure()
        columns = 1
        rows = num_rotations
        count = 1

    for i in range(K_sequence.shape[0]):
        cam = Camera(0, K_sequence[i], np.diag([1, 1, 1]), np.zeros((3, 1)))
        # apply random rotation using lie groups...
        rand = np.random.rand(3)
        Rot = expm(skew(rand))

        # generate random translation

        trans = np.random.rand(3)

        Rot = np.vstack((Rot, trans))
        Rot = np.hstack((Rot, np.zeros((4, 1))))
        Rot[-1, -1] = 1

        object_points_copy = (Rot @ object_points.T).T

        points_2d[i, :] = cam.project(object_points_copy.T, generate_noise, sigma).T

        if plot:
            ax = fig.add_subplot(rows, columns, count)
            ax.set_title(f"view from: {i}")
            ax.set(xlim=(IM_WIDTH, 0), ylim=(0, IM_HEIGHT))

            colors = np.arange(0, object_points.shape[0])
            ax.scatter(points_2d[i, :][:, 0], points_2d[i, :][:, 1], c=colors, cmap="autumn")
            count += 1
            plt.draw()

            three_d_fig = plt.figure()
            three_d_ax = plt.axes(projection="3d")

            plot_cam_axis(cam, three_d_ax)

            three_d_ax.scatter(
                object_points_copy[:, 0],
                object_points_copy[:, 1],
                object_points_copy[:, 2],
                c=colors,
                cmap="autumn")
            three_d_ax.view_init(elev=-90, azim=90)
            set_aspect_equal_3d(three_d_ax)

            plt.ion()

    if plot:
        plt.show()

    visibilityMat = np.full((points_2d.shape[0], points_2d.shape[1]), True)

    # for row in range(num_rotations):
    #     for col, point in enumerate(points_2d[row]):
    #         if all(point[0:2] >= 0) and point[0] < IM_WIDTH and point[1] < IM_HEIGHT:
    #             visibilityMat[row, col] = True
    #         else:
    #             visibilityMat[row, col] = False

    pointsA, pointsB = getPointsSharedBetweenViews(points_2d[0], points_2d[1], visibilityMat[0],
                                                   visibilityMat[1])
    F12 = computeFundamentalMatrixBetween(pointsA, pointsB)

    Ps = np.zeros((num_rotations, 3, 4))
    P1, P2 = createProjectivelyAmbiguousCameraMatrices(F12)
    Ps[0, :, :] = P1
    Ps[1, :, :] = P2

    points3D = np.full((points_2d[0].shape[0], 4), -1.0)

    initialStructure = triangulatePoints(P1, P2, points_2d[:2], 0, 1, visibilityMat)
    points3D[:, :] = initialStructure

    # # at this point at view N, I should have access to N 3D points and the projection of those points in view N
    cams = [0, 1]
    for i in range(2, num_rotations):
        cams.append(i)
        Pi = resectionCameraWith(points3D, points_2d[i], i, visibilityMat, cleanIputs=False)
        Ps[i, :, :] = Pi

    Ks, isDegenerate, maxDisturbed = self_calibrate(Ps, IM_WIDTH, IM_HEIGHT)

    # report the ratio between estimated and ground truth focal length.
    K_diffs = np.zeros((len(Ks), 2), dtype=np.float64)
    for i, K_mat in enumerate(Ks):
        computed_fx = K_mat[0, 0]
        computed_fy = K_mat[1, 1]

        actual_fx = K_sequence[i][0, 0]
        actual_fy = K_sequence[i][1, 1]

        difference_fx = math.fabs(computed_fx - actual_fx) / actual_fx
        difference_fy = math.fabs(computed_fy - actual_fy) / actual_fy

        K_diffs[i, 0] = difference_fx
        K_diffs[i, 1] = difference_fy

    Average_K_Error = np.sum(K_diffs, axis=0)
    Average_K_Error /= K_diffs.shape[0]

    return Average_K_Error, isDegenerate, maxDisturbed


if __name__ == '__main__':
    num_trials = 500
    num_varying_sigma = 1
    num_views = 6
    static_f = True
    generate_noise = True
    plot = False

    if plot:
        import matplotlib as mpl
        mpl.use('Qt4Agg')

    errors_fx = []
    errors_fy = []
    sigmas = []
    runs = 0
    sumDisturbed = 0
    degenCount = 0
    for i in range(num_varying_sigma):
        sigma = 0 + i * 0.02

        w = IM_WIDTH  # uniform(900, 1000)
        h = IM_HEIGHT  # uniform(900, 1000)
        if static_f:
            f1 = uniform(800, 1200)
            f2 = f1

        K_sequence = np.zeros((num_views, 3, 3), dtype=float)
        for i in range(num_views):

            # assume fx = fy
            if not static_f:
                f1 = uniform(800, 1200)
                f2 = f1

            K_sequence[i, :, :] = np.array([[f1, 0, (w - 1) / 2], [0, f2, (h - 1) / 2], [0, 0, 1]],
                                           dtype=np.float32)

        results = np.zeros((num_trials, 2), dtype=float)
        for trial in range(num_trials):
            result, isDegenerate, maxDisturbed = run_experiment(K_sequence, generate_noise, sigma, plot=plot)
            runs += 1
            sumDisturbed += maxDisturbed
            if isDegenerate:
                degenCount += 1

            results[trial, :] = result

        # print(f"For {num_trials} trials with {num_views} views each and sigma {sigma}:")
        # print(f"Is using same focal lengths per trial: {static_f}")
        net_error = np.sum(results, axis=0)
        net_error /= num_trials
        # print(f"\tAverage Error for f_x: {net_error[0]}")
        # print(f"\tAverage Error for f_y: {net_error[1]}")
        sigmas.append(sigma)
        errors_fx.append(net_error[0] * 100)
        errors_fy.append(net_error[1] * 100)

    print(f"\tDegeneracy average disturbed: {sumDisturbed / runs}")
    print(f"\tDegeneracy Percentage for {runs}: {(degenCount / runs) * 100}")

    # now plot these bois
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set(ylim=(0, 100))
    major_ticks = np.arange(0, 101, 10)
    minor_ticks = np.arange(0, 101, 5)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')

    plt.title(f'Average Error of {num_trials} trials for {num_views} views with Gaussian Noise')
    plt.plot(sigmas, errors_fx, linestyle='--', marker='o', color='b')
    plt.plot(sigmas, errors_fy, linestyle='-', marker='d', color='g')
    plt.xlabel('sigma', fontsize=12)
    plt.ylabel('error (%)', fontsize=12)
    plt.legend(['error fx', 'error fy'], loc='upper left')
    plt.show()
