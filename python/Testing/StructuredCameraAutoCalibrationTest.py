# This file is going to test how well this auto calibration works for dome configurations with a certain shape at the origin....
import numpy as np
import math
from math import pi as PI

from python.AutoCalibration.projectiveReconstruct import *
from python.AutoCalibration.flyingThingsUtils import *
from python.AutoCalibration.calibrationAPI import self_calibrate

from python.ParametricObjects.ParametricDome import ParametricDome as Dome
from python.ParametricObjects.ParametricRandomPointCloud import ParametricRandomPointCloud
from python.AutoCalibration.Camera import Camera
from python.ParametricObjects.ParametricCube import ParametricCube
import mpl_toolkits.mplot3d as plt3d
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


def projectPoint(P, Xi, im_height, im_width):
    projection = P @ Xi
    projection /= projection[2]
    return projection


def projectPointsWithCam(P, X, im_height, im_width):
    points = np.ones((X.shape[0], 3))
    for i, Xi in enumerate(X):
        points[i, :] = projectPoint(P, Xi, im_height, im_width)
    return points


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


def run_experiment(static_f, generate_noise, sigma, plot=False):
    minCube = np.array([-2, -2.5, -1])
    maxCube = np.array([2, 6, 1.5])

    dome = Dome(3, 5, 3)

    worldUp = normalize_vector(np.array([0.0, 0.0, 1.0]))
    lookAtPos = (minCube + maxCube) / 2
    steps_u = 2
    steps_v = 3
    num_views = steps_u * steps_v
    us = np.linspace(0, 2 * PI, steps_u, False, dtype=np.float)
    vs = np.linspace(PI / 18, PI / 2, steps_v, False, dtype=np.float)
    us2, vs2 = np.meshgrid(us, vs)
    cameras = []
    count = 0

    if static_f:
        f1 = uniform(800, 1200)
        f2 = f1

    w = IM_WIDTH  # uniform(300,1000)
    h = IM_HEIGHT  # uniform(300,1000)

    for i in range(len(us)):
        for j in range(len(vs)):
            u = us2[j, i]
            v = vs2[j, i]
            # In world coordinates...
            domePos = dome.generatePoint(u, v)

            forwardVec = normalize_vector(lookAtPos - domePos)
            if np.all(forwardVec == 0):
                forwardVec = np.array([0, 0, 1])

            right = normalize_vector(np.cross(worldUp, forwardVec))
            if np.all(right == 0):
                right = np.array([1, 0, 0])
            up = normalize_vector(np.cross(forwardVec, right))

            # camera is defined as P = KR[I | -C] OR P = K[R|t] where t = -RC
            # C is the world coordinate of the point.
            Rc = np.zeros((3, 3), dtype=np.float32)
            Rc[:, 0] = right
            Rc[:, 1] = up
            Rc[:, 2] = forwardVec

            # # apply a random offset to this rotation about this axis?
            randpsi = np.random.uniform(-1, 1, 3)
            randRot = skew(randpsi)
            rotationIn = expm(randRot)

            Rc = Rc @ rotationIn

            # verify this is a valid member of SO(3) a.k.a rotation matrices.
            assert (1 - np.linalg.det(Rc) <= 1e-5)
            assert (np.all(np.abs(Rc.T @ Rc - np.eye(3)) < 1e-5))

            # assume fx = fy
            if not static_f:
                f1 = uniform(800, 1200)
                f2 = f1

            k = np.array([[f1, 0, (w - 1) / 2], [0, f2, (h - 1) / 2], [0, 0, 1]], dtype=np.float32)

            camera = Camera(count, k, Rc.T, -Rc.T @ domePos.reshape((3, 1)))
            cameras.append(camera)
            count += 1

    num_views = count

    if plot:
        import matplotlib as mpl
        mpl.use('Qt4Agg')

        three_d_fig = plt.figure()
        three_d_ax = plt.axes(projection="3d")

    cube = ParametricCube(minCube, maxCube)

    steps_cube_u = 5
    steps_cube_v = 5

    planeU = np.linspace(0.2, 1, steps_cube_u, False, dtype=np.float)
    planeV = np.linspace(0.2, 1, steps_cube_v, False, dtype=np.float)
    planeUs, planeVs = np.meshgrid(planeU, planeV)

    num_points = 30
    object_points = []
    # cloud = ParametricRandomPointCloud(maxCube, minCube)
    # for i in range(num_points):
    #     cloud_point = cloud.generatePoint()
    #     object_points.append(cloud_point)

    for faceW in range(1):
        for uRow in planeUs:
            for vRow in planeVs:

                for k in range(len(uRow)):
                    u = uRow[k]
                    v = vRow[k]
                    point = cube.generatePoint(u, v, faceW)
                    object_points.append(point)

    object_points = np.array(object_points)

    # now need to project using these extrinsics and points...
    object_points = np.hstack((object_points, np.ones((object_points.shape[0], 1))))

    points_2d = np.ones((len(cameras), object_points.shape[0], 3))

    if plot:
        fig = plt.figure()
        columns = 1
        rows = (num_views // 1) + 1
        count = 1

    for view_index, cam in enumerate(cameras):
        R = cam.R.T
        cam_position = -(R @ cam.t).reshape(3)

        points_2d[view_index, :] = cam.project(object_points.T, generate_noise, sigma).T

        if plot:
            three_d_ax.scatter([cam_position[0]], [cam_position[1]], [cam_position[2]],
                               c=[cam_position[2]],
                               cmap="rainbow")

            #plot x,y,z axis...
            length = 0.5
            x_line = cam_position + R[:, 0] * length  # first basis vector..
            y_line = cam_position + R[:, 1] * length  # second basis vector..
            z_line = cam_position + R[:, 2] * length  # third basis vector..

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

            # draw something along the -z axis...

            # plot the projected points...
            ax = fig.add_subplot(rows, columns, count)
            ax.set_title(f"view from: {view_index}")

            ax.set(xlim=(IM_WIDTH, 0), ylim=(0, IM_HEIGHT))
            colors = np.arange(0, object_points.shape[0])
            ax.scatter(
                points_2d[view_index, :][:, 0], points_2d[view_index, :][:, 1], c=colors, cmap="autumn")
            count += 1

    # ax.set_aspect('equal')
    if plot:
        colors = np.arange(0, object_points.shape[0])
        three_d_ax.scatter(
            object_points[:, 0], object_points[:, 1], object_points[:, 2], c=colors, cmap="autumn")
        three_d_ax.view_init(elev=-80, azim=90)
        set_aspect_equal_3d(three_d_ax)
        plt.ion()
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

    Ps = np.zeros((len(cameras), 3, 4))
    P1, P2 = createProjectivelyAmbiguousCameraMatrices(F12)
    Ps[0, :, :] = P1
    Ps[1, :, :] = P2

    points3D = np.full((points_2d[0].shape[0], 4), -1.0)

    initialStructure = triangulatePoints(P1, P2, points_2d[:2], 0, 1, visibilityMat)
    points3D[:, :] = initialStructure

    points3D = np.full((points_2d[0].shape[0], 4), -1.0)

    initialStructure = triangulatePoints(P1, P2, points_2d[:2], 0, 1, visibilityMat)
    points3D[:, :] = initialStructure

    # # at this point at view N, I should have access to N 3D points and the projection of those points in view N
    cams = [0, 1]
    for i in range(2, len(cameras)):
        cams.append(i)
        Pi = resectionCameraWith(points3D, points_2d[i], i, visibilityMat, cleanIputs=False)
        Ps[i, :, :] = Pi

    error = simpleReprojectionError(points3D, Ps, points_2d, cams, visibilityMat)
    print(f"Reprojection Error: {error}")

    Ks, isDegenerate, maxDisturbed = self_calibrate(Ps, IM_WIDTH, IM_HEIGHT)

    # report the ratio between estimated and ground truth focal length.
    K_diffs = np.zeros((len(Ks), 2), dtype=np.float64)
    for i, K_mat in enumerate(Ks):
        cam = cameras[i]
        # print(f"Computed Intrinsic:\n{K_mat}")
        # print(f"Actual Intrinsic:\n{cam.K}")
        # if isDegenerate:
        #     print(f"Computed Intrinsic:\n{K_mat}")
        #     print(f"Actual Intrinsic:\n{cam.K}")
        computed_fx = K_mat[0, 0]
        computed_fy = K_mat[1, 1]

        computed_s = K_mat[0, 1]

        if abs(computed_s) > 10.0:
            isDegenerate = True

        actual_fx = cam.K[0, 0]
        actual_fy = cam.K[1, 1]

        difference_fx = math.fabs(computed_fx - actual_fx) / actual_fx
        difference_fy = math.fabs(computed_fy - actual_fy) / actual_fy

        K_diffs[i, 0] = difference_fx
        K_diffs[i, 1] = difference_fy

    Average_K_Error = np.sum(K_diffs, axis=0)
    Average_K_Error /= K_diffs.shape[0]

    #print(Average_K_Error)

    return Average_K_Error, K_diffs.shape[0], isDegenerate, maxDisturbed


def main():
    enable_noise = False
    static_f = True
    num_times_per_sigma = 1
    num_runs = 500
    plot = False

    errors_fx = []
    errors_fy = []
    sigmas = []

    runs = 0
    degenCount = 0
    sumDisturbed = 0
    for i in range(num_runs):
        sigma = 0 + i * 0.2

        results = np.zeros((num_times_per_sigma, 2), dtype=float)
        for j in range(num_times_per_sigma):
            error, num_views, degenerate, maxDisturbed = run_experiment(
                static_f, enable_noise, sigma, plot=plot)
            runs += 1
            sumDisturbed += maxDisturbed
            if degenerate:
                degenCount += 1
            results[j, :] = error

        net_error = np.sum(results, axis=0)
        net_error /= num_times_per_sigma
        #print(f"Noise added with std dev: {sigma}")
        #print(f"\tAverage Error for f_x: {error[0] * 100}%")
        #print(f"\tAverage Error for f_y: {error[1] * 100}%")

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

    plt.title(f'Error for {num_times_per_sigma} trials of {num_views} views with varying Gaussian Noise')
    plt.plot(sigmas, errors_fx, linestyle='--', marker='o', color='b')
    plt.plot(sigmas, errors_fy, linestyle='-', marker='d', color='g')
    plt.xlabel('sigma', fontsize=12)
    plt.ylabel('error (%)', fontsize=12)
    plt.legend(['error fx', 'error fy'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
