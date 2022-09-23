import numpy as np
from scipy.linalg import null_space
np.set_printoptions(precision=3, suppress=True)
import time
from python.AutoCalibration.flyingThingsUtils import *
from python.AutoCalibration.DLT import *


def dt(t0):
    return round(time.time() - t0, 2)


# epipole in left image is F*e(l) = 0.. nullspace of F* e(l) = 0
# epipole in the right images if (e(r)^T * F)^T =    F^T * e(r) = 0


def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    """ img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r = img1.shape[0]
    c = img1.shape[1]
    try:
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(np.copy(img1), (x0, y0), (x1, y1), color, 1)
        return img1, img2
    except:
        return img1, img2


def draw_epilines_and_points(frame1, frame2, F, points1, points2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(points2, 2, F)

    lines1 = lines1.reshape(-1, 3)
    img5, img6 = draw_epipolar_lines(frame1, frame2, lines1, points1, points2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(points1, 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = draw_epipolar_lines(frame2, frame1, lines2, points2, points1)

    draw_points_in_image(points1, img5)
    draw_points_in_image(points2, img3)

    return img5, img3


def computeEpipole(F):
    """ Computes the (right) epipole from a
           fundamental matrix F.
           (Use with F.T for left epipole.) """

    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F, full_matrices=True)
    e = V[-1]
    return e / (e[2] * 1.0)


def create_projectively_ambiguous_camera_matrices(F):
    if F is None:
        print(f"not possible")
    e1 = computeEpipole(F)
    e2 = computeEpipole(F.T)
    e2SkewMat = skew(e2)

    P = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
    Pp = np.concatenate((e2SkewMat @ F, e2.reshape((3, 1))), axis=1)
    return P, Pp


def compute_fundamental_matrix_between(points1, points2):
    #Fres, mask = cv2.findFundamentalMat(pts_1, pts_2, cv2.FM_8POINT, 3, 0.99)
    Fres, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1, 0.9)

    return Fres, mask


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)


# reprojection error is given by xij = PiXj normalized for xij[2] = 1
def single_simple_reprojection_error(points3D, cameraMatrix, points2D, view, visibilityMat):
    """
    :param view: view in question
    :param points3D: Points in 3D for the image points
    :param cameraMatrix: Camera Matrix for the points
    :param points2D: Points observed on 2D image
    :return:
    """
    projPoints = np.full((points3D.shape[0], 3), float('nan'))
    for i, pt in enumerate(points3D):
        if visibilityMat[view, i] and not np.all(np.isnan(pt)):
            projection = cameraMatrix @ pt
            projection = projection / projection[2]
            projPoints[i, :] = projection

    totalError = 0
    count = 0
    for i, info in enumerate(zip(projPoints, points2D)):
        reprojected = info[0]
        original = info[1]
        if visibilityMat[view, i] and not np.all(np.isnan(original)) and not np.all(np.isnan(reprojected)):
            #print(f"\treprojct: {reprojected}\t||\toriginal:{original}")
            dist = np.linalg.norm(reprojected - original)**2
            #print (f"\tdist: {dist} for {i}")
            totalError += dist
            count += 1

    return totalError


def simple_reprojection_error(points3D, cameraMatrices, points2D, views, visibilityMat):
    """
    :param points3D: 3D points
    :param cameraMatrices: The camera matrices for the error calculation
    :param points2D: array of points in image coordiantes (x,y,1)
    :param views: keys for the views.
    :param visibilityMat: matrix that says if a point j is visible in view i, (i,j)
    :return total error: reprojection error.
    """
    totalError = 0.0
    camCount = 0
    for i, cam in enumerate(cameraMatrices):
        #print(f"Reprojection Error for Camera:{views[i]}\nP:{cam}")
        points2DForView = points2D[views[i], :]
        totalError += single_simple_reprojection_error(points3D, cam, points2DForView, i, visibilityMat)
        camCount += 1

    return totalError


def triangulate_point(p1, p2, point1, point2, T1, T2):
    A = np.zeros((4, 4))
    u1p = T1 @ point1
    u2p = T2 @ point2

    p1p = T1 @ p1
    p2P = T2 @ p2

    A[0, :] = u1p[1] * p1p[2, :] - p1p[1, :]
    A[1, :] = p1p[0, :] - u1p[0] * p1p[2, :]

    A[2, :] = u2p[1] * p2P[2, :] - p2P[1, :]
    A[3, :] = p2P[0, :] - u2p[0] * p2P[2, :]

    U, s, V = np.linalg.svd(A)
    V = np.transpose(V)

    pt_3D = V[:, 3] / V[3, 3]

    return pt_3D


def triangulate_points(p1, p2, points2d, view1, view2, visibilityMatrix):
    # TODO maybe change this to use the visibility matrix to run this computation...

    points12d = create_array_of_same_size_cleaned(points2d[view1], view1, visibilityMatrix)
    points22d = create_array_of_same_size_cleaned(points2d[view2], view2, visibilityMatrix)

    T1 = isotropic_2D_scaling_transformation_for(points2d[view1], view1, visibilityMatrix)
    T2 = isotropic_2D_scaling_transformation_for(points2d[view2], view2, visibilityMatrix)

    points3D = np.full((points2d[0].shape[0], 4), 1.0)
    for i, points in enumerate(zip(points12d, points22d)):
        if visibilityMatrix[view1, i] and visibilityMatrix[view2, i]:
            point3D = triangulate_point(p1, p2, points[view1], points[view2], T1, T2)
            points3D[i, :3] = point3D
        else:
            points3D[i, :] = np.array([None, None, None, None])

    return points3D


def triangulate_points_no_visibility_matrix(P1, P2, points_2d) -> np.ndarray:
    """
    :param P1: Projection Matrix 1 ndarray
    :param P2: Projection Matrix 2 ndarray
    :param points_2d: a tensor of size Num_Points x 2 x 3 that is each u,v observation
    :return:
    """
    # assume these points are clean no need to check visibility matrices or anything here.

    uv_points_1 = points_2d[0]
    uv_points_2 = points_2d[1]

    assert (uv_points_1.shape[0] == uv_points_2.shape[0])

    T_Norm_1 = isotropic_2D_scaling_transform(uv_points_1)
    T_Norm_2 = isotropic_2D_scaling_transform(uv_points_2)

    triangulated_3D_points = np.full((uv_points_1.shape[0], 4), 1.0)

    for i, points in enumerate(zip(uv_points_1, uv_points_2)):
        point_3d = triangulate_point(P1, P2, points[0], points[1], T_Norm_1, T_Norm_2)
        triangulated_3D_points[i, :] = point_3d

    return triangulated_3D_points


def cleanInputsToDLT(points3D, points2D, view, visibilityMat):
    clean3D = []
    clean2D = []
    for i, entry in enumerate(zip(points3D, points2D)):
        point3D = entry[0]
        point2D = entry[1]
        if not np.all(np.isnan(point3D)) and np.all(
                point2D >= 0) and not np.any(np.isnan(point2D)) and visibilityMat[view, i]:
            clean3D.append(point3D)
            clean2D.append(point2D)

    clean3D = np.array(clean3D)
    clean2D = np.array(clean2D)

    return clean3D, clean2D


def resection_camera_with(points3D, points2D, view, visibilityMatrix, cleanIputs=True):
    if cleanIputs:
        p3d, p2d = cleanInputsToDLT(points3D, points2D, view, visibilityMatrix)
        P = DLTcalib(p3d, p2d, view, visibilityMatrix)
        return P.reshape((3, 4))
    else:
        P = DLTcalib(points3D, points2D, view, visibilityMatrix)
        return P.reshape((3, 4))


def get_point_mask_for_visibility_between_views(points_uv_1, points_uv_2, visibility_points_uv1,
                                                visibility_points_uv2):
    # have to be same length (finite amount of points measured)
    assert (points_uv_1.shape[0] == points_uv_2.shape[0])
    mask = []
    for i, points_uv in enumerate(zip(points_uv_1, points_uv_2)):
        is_in_one = visibility_points_uv1[i]
        is_in_two = visibility_points_uv2[i]

        if is_in_one and is_in_two:
            mask.append(1)
        else:
            mask.append(0)
    return np.array(mask).reshape(-1, 1)


def get_points_shared_between_views(points_uv_1, points_uv_2, visibility_points_uv1, visibility_points_uv2):

    point_mask = get_point_mask_for_visibility_between_views(points_uv_1, points_uv_2, visibility_points_uv1,
                                                             visibility_points_uv2)
    points_uv_1_good = filter_points_using_mask(points_uv_1, point_mask)
    points_uv_2_good = filter_points_using_mask(points_uv_2, point_mask)

    assert (points_uv_1_good.shape[0] == points_uv_2_good.shape[0])
    return points_uv_1_good, points_uv_2_good


def filter_points_using_mask(points_uv, point_mask):
    return points_uv[point_mask.ravel() == 1]
