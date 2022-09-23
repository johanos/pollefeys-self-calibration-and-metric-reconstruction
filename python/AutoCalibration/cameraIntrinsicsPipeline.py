# The point of this file is to
# 1. Create a Projective Reconstruction of a series of images
# 2. Auto Calibrate that Projective Reconstruction using the Dual Image of the Absolute Quadric.

# START IMPORTS #
import matplotlib.pyplot as plt

from python.AutoCalibration.Camera import Camera
from python.AutoCalibration.calibrationAPI import self_calibrate, flyingThingsIntrinsicMatrix
from python.AutoCalibration.ProjectiveReconstructionHelpers import *
from python.AutoCalibration.flyingThingsUtils import *
import numpy as np

from python.PointCorrespondances.FlyingThings import generatePointTrajectories as generatePointsFT
from python.PointCorrespondances.Sintel import generatePointTrajectories as generatePointsSINTEL


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

            projection = cam.project(point_3d.reshape(4, 1), False, 0).reshape(3)
            # calculate the distance between projection and the point uv
            point_uv = points_uv_for_view[j]

            squared_distance = np.linalg.norm(projection - point_uv)**2
            total_error += squared_distance
            total_entries += 1

    return total_error, total_entries


# END IMPORTS

# number_frames = 4
# start_frame = 25

number_frames = 5
start_frame = 6
# 25 and 5 gives good result right now... alley_2
# 25 and 4 for cave_2

print("---" * 40)
print("\tRunning Pipeline")
print("\n" + "--" * 8 + " Stage 1: Point Correspondences " + "--" * 8)

points, images, gt_intrinsic = generatePointsFT("C/0010", number_frames, start_frame)
#points, images, gt_intrinsic = generatePointsSINTEL("alley_2", number_frames, start_frame)

imHeight = images[0].shape[0]
imWidth = images[0].shape[1]

visibilityMat = np.full((points.shape[0], points.shape[1]), False)

for row in range(number_frames):
    for col, point in enumerate(points[row]):
        if all(point[0:2] >= 0) and point[0] < imWidth and point[1] < imHeight:
            visibilityMat[row, col] = True
        else:
            visibilityMat[row, col] = False

print("--" * 8 + f" Computed Points from: {start_frame} to: {start_frame + number_frames} " + "--" * 8)
print(f'\tAlso Computed V2P Matrix of shape\n:{visibilityMat.shape}')

print("\n" + "--" * 8 + " Stage 2: Compute Fundamental Matrix between view 1 and view 2 " + "--" * 8)
print("---" * 40)
print("\tComputing Fundamental Matrix for First View and Second View to Kick Off Projective Reconstruction")

print(f"\tGet Frame 1 and Frame 2")
frame1 = images[0]
frame2 = images[1]

print("\tComputing Fundamental Matrix With 8 Points Algorithm")

pointsA, pointsB = get_points_shared_between_views(points[0], points[1], visibilityMat[0], visibilityMat[1])
print(
    f"\tPointsA and PointsB have shape:\n\t{pointsA.shape}\n\twhereas points[0] and points[1] have shape:\n\t{points.shape}"
)

F12, mask = compute_fundamental_matrix_between(pointsA, pointsB)
#Fs[1, :, :] = F
points_0 = points[0][mask.ravel() == 1]
points_1 = points[1][mask.ravel() == 1]
imL, imR = draw_epilines_and_points(frame1, frame2, F12, points_0, points_1)

print("---" * 40)
epipolar_ims = [imL, imR]

fig = plt.figure(figsize=(10, 10))
columns = 2
rows = (len(epipolar_ims) // 2) + 1
count = 1
for i in range(0, len(epipolar_ims), 2):
    img = epipolar_ims[i]
    img2 = epipolar_ims[i + 1]
    ax = fig.add_subplot(rows, columns, count)
    ax.set_title(f"Frame {i} and Frame {i // 2 + 1}")
    plt.imshow(img)
    count += 1
    ax2 = fig.add_subplot(rows, columns, count)
    ax2.set_title(f"Frame {i} and Frame {i // 2 + 1}")
    plt.imshow(img2)
    count += 1
plt.show()

print("\n" + "--" * 8 + " Stage 3: Compute Initial Projection Matrices Using F12 " + "--" * 8)

Ps = np.zeros((number_frames, 3, 4))

P1, P2 = create_projectively_ambiguous_camera_matrices(F12)

Ps[0, :, :] = P1
print(f"\nCAMERA 1:")
print(f"\t{P1[0, :]}")
print(f"\t{P1[1, :]}")
print(f"\t{P1[2, :]}")

Ps[1, :, :] = P2
print(f"\nCAMERA 2:")
print(f"\t{P2[0, :]}")
print(f"\t{P2[1, :]}")
print(f"\t{P2[2, :]}")

print("\n" + "--" * 7 + " Stage 4: Compute Initial Reconstruction using P0, P1 " + "--" * 7)

# these are my points...

points3D = np.full((points[0].shape[0], 4), -1.0)

# would need to clean up the correpsondences here

points_uv_0, points_uv_1 = get_points_shared_between_views(points[0], points[1], visibilityMat[0],
                                                           visibilityMat[1])
combined_uv = np.array([points_uv_0, points_uv_1])
initialStructure = triangulate_points_no_visibility_matrix(P1, P2, combined_uv)
#initialStructure = triangulate_points(P1, P2, points[:2], 0, 1, visibilityMat)
points3D[:, :] = initialStructure

P0_new = Camera(0, None, None, None, P1)
P1_new = Camera(0, None, None, None, P2)

cams_to_use = [P0_new, P1_new]
points_to_use = np.array([points_uv_0, points_uv_1])

error = simple_reprojection_error(points3D, np.array([P1, P2]), points[:2], [0, 1], visibilityMat)
error_new, total = calculate_reprojection_error(initialStructure, cams_to_use, points_to_use, visibilityMat)

print(f"Reprojection error: {error}")
print(f"Reproj: {error_new}")
print(f"average: {(error_new/total):4f}")
print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

# GOAL ONE ACHIEVED, LOW REPROJECTION ERROR ON PERFECT POINTS
print("\n" + "--" * 6 + " Stage 5: Resection New View using Initial Reconstruction Frame to Compute PN " +
      "--" * 6)

cams = [0, 1]
# # at this point at view N, I should have access to N 3D points and the projection of those points in view N
for i in range(2, number_frames):
    #pointsA, pointsB = cleanUpPointCorrespondances(normalizedPoints[0], normalizedPoints[i], v2pMatrix[0], v2pMatrix[i])
    cams.append(i)
    Pi = resection_camera_with(points3D, points[i], i, visibilityMat)
    Ps[i, :, :] = Pi

# now I should see the reprojection error
error = simple_reprojection_error(points3D, Ps, points, cams, visibilityMat)

print(f"Composite Reprojection error: {error}")
print(f"\nIf the above are low, then we reasonably computed a projective frame for these points")

print(f"Ground Truth Intrinsic Matrix:\n{gt_intrinsic[0]}")
Ks, isDegenerate, maxDisturbed = self_calibrate(Ps, imWidth, imHeight)

print(f"Computed Intrinsic Matrix:\n{Ks[0]}")
print(f"Is this degenerate or close to degenerate?: {isDegenerate}")

K_diffs = np.zeros((len(Ks), 2), dtype=np.float64)
for i in range(len(Ks)):
    K = Ks[i]
    computed_fx = K[0, 0]
    computed_fy = K[1, 1]

    actual_fx = gt_intrinsic[i][0, 0]
    actual_fy = gt_intrinsic[i][1, 1]

    difference_fx = math.fabs(computed_fx - actual_fx) / actual_fx
    difference_fy = math.fabs(computed_fy - actual_fy) / actual_fy

    K_diffs[i, 0] = difference_fx
    K_diffs[i, 1] = difference_fy

Average_K_Error = np.sum(K_diffs, axis=0)
Average_K_Error /= K_diffs.shape[0]

print(f"Average Error Fx: {Average_K_Error[0] * 100}%")
print(f"Average Error Fy: {Average_K_Error[1] * 100}%")
