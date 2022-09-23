# The point of this file is to
# 1. Create a Projective Reconstruction of a series of images
# 2. Auto Calibrate that Projective Reconstruction using the Dual Image of the Absolute Quadric.

# START IMPORTS #
import matplotlib.pyplot as plt
from python.AutoCalibration.calibrationAPI import self_calibrate
from python.AutoCalibration.ProjectiveReconstructionHelpers import *
from python.AutoCalibration.flyingThingsUtils import *
from python.AutoCalibration.readPfm import readPFM
import numpy as np

# END IMPORTS #


def main():
    numFrames = 2
    startFrame = 6
    frameNames = [""] * numFrames
    for frame in range(numFrames):
        frameNames[frame] = str(startFrame + frame).zfill(4)
    sequence = "C/0144"
    print("---" * 40)
    print("\tRunning Pipeline")
    print("\n" + "--" * 8 + " Stage 1: Point Correspondences " + "--" * 8)

    frameTemplate = "TestingData/frames/TEST/{}/left/{}.png"
    frameOFTemplate = "TestingData/optical_flow/TEST/{}/into_future/left/OpticalFlowIntoFuture_{}_L.pfm"
    pointTemplate = "TestingData/frames/TEST/{}/left/pt_2D_{}.txt"
    points1txt = pointTemplate.format(sequence, startFrame)

    points = np.array([(getDataFromTxtFile(points1txt))], dtype=np.float)
    #Fs = np.zeros((numFrames, 3, 3))

    images = []
    print("---" * 40)
    print("\tComputing Points For Sequence")
    for i in range(0, numFrames - 1):
        nextOFfile = frameOFTemplate.format(sequence, str(startFrame + i).zfill(4))
        nextOF, scale = readPFM(nextOFfile)
        print(f"\tReading Optical Flow File {nextOFfile}")
        temp = computeNewPositions(points[i], nextOF)
        points = np.append(points, [temp], axis=0)

    frame1File = frameTemplate.format(sequence, str(startFrame).zfill(4))
    frame1 = plt.imread(frame1File)
    imHeight = frame1.shape[0]
    imWidth = frame1.shape[1]

    visibilityMat = np.zeros((points.shape[0], points.shape[1]))

    for i in range(numFrames):
        for j, point in enumerate(points[i]):
            if all(point[0:2] >= 0) and point[0] < imWidth and point[1] < imHeight:
                visibilityMat[i, j] = True
            else:
                visibilityMat[i, j] = False

    print("--" * 8 + f" Computed Points from: {startFrame} to: {startFrame+numFrames} " + "--" * 8)
    print(f'\tAlso Computed V2P Matrix of shape\n:{visibilityMat.shape}')

    print("\n" + "--" * 8 + " Stage 2: Compute Fundamental Matrix between view 1 and view 2 " + "--" * 8)
    print("---" * 40)
    print(
        "\tComputing Fundamental Matrix for First View and Second View to Kick Off Projective Reconstruction")

    print(f"\tRead Frame 1: {frame1File}")

    frame2File = frameTemplate.format(sequence, str(startFrame + 1).zfill(4))
    frame2 = plt.imread(frame2File)
    print(f"\tReading Frame 2: {frame2File}")

    print("\tComputing Fundamental Matrix With 8 Points Algorithm")

    points1, points2 = getPointsSharedBetweenViews(points[0], points[1], visibilityMat[0], visibilityMat[1])
    print(
        f"\tPoints1 and PointsB have shape:\n\t{points1.shape}\n\twhereas points[0] and points[1] have shape:\n\t{points.shape}"
    )

    F12 = computeFundamentalMatrixBetween(points1, points2)
    #Fs[1, :, :] = F
    imL, imR = draw_epilines_and_points(frame1, frame2, F12, points[0], points[1])

    print("---" * 40)
    images.extend([imL, imR])

    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = (len(images) // 2) + 1
    count = 1
    for i in range(0, len(images), 2):
        img = images[i]
        img2 = images[i + 1]
        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(f"Frame {frameNames[0]} and Frame {frameNames[i // 2 + 1]}")
        plt.imshow(img)
        count += 1
        ax2 = fig.add_subplot(rows, columns, count)
        ax2.set_title(f"Frame {frameNames[0]} and Frame {frameNames[i // 2 + 1]}")
        plt.imshow(img2)
        count += 1
    plt.show()

    print("\n" + "--" * 8 + " Stage 3: Compute Initial Projection Matrices Using F12 " + "--" * 8)

    Ps = np.zeros((numFrames, 3, 4))
    P1, P2 = createProjectivelyAmbiguousCameraMatrices(points1, points2, F12)

    # points1, points3 = getPointsSharedBetweenViews(points[0], points[2], visibilityMat[0], visibilityMat[2])
    # F13 = computeFundamentalMatrixBetween(points1, points3)
    # _ , P3 = createProjectivelyAmbiguousCameraMatrices(points1, points3, F13)
    #
    # points1, points4 = getPointsSharedBetweenViews(points[0], points[3], visibilityMat[0], visibilityMat[3])
    # F14 = computeFundamentalMatrixBetween(points1, points4)
    # _, P4 = createProjectivelyAmbiguousCameraMatrices(points1, points4, F14)
    #
    # points1, points5 = getPointsSharedBetweenViews(points[0], points[4], visibilityMat[0], visibilityMat[4])
    # F15 = computeFundamentalMatrixBetween(points1, points5)
    # _, P5 = createProjectivelyAmbiguousCameraMatrices(points1, points5, F15)
    #
    # points1, points6 = getPointsSharedBetweenViews(points[0], points[5], visibilityMat[0], visibilityMat[5])
    # F16 = computeFundamentalMatrixBetween(points1, points6)
    # _, P6 = createProjectivelyAmbiguousCameraMatrices(points1, points6, F16)
    #
    # points1, points7 = getPointsSharedBetweenViews(points[0], points[6], visibilityMat[0], visibilityMat[6])
    # F17 = computeFundamentalMatrixBetween(points1, points7)
    # _, P7 = createProjectivelyAmbiguousCameraMatrices(points1, points7, F17)

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

    # Ps[2, :, :] = P3
    # print(f"\nCAMERA 3:")
    # print(f"\t{P3[0, :]}")
    # print(f"\t{P3[1, :]}")
    # print(f"\t{P3[2, :]}")
    #
    # Ps[3, :, :] = P4
    # print(f"\nCAMERA 4:")
    # print(f"\t{P4[0, :]}")
    # print(f"\t{P4[1, :]}")
    # print(f"\t{P4[2, :]}")
    #
    # Ps[4, :, :] = P5
    # print(f"\nCAMERA 5:")
    # print(f"\t{P5[0, :]}")
    # print(f"\t{P5[1, :]}")
    # print(f"\t{P5[2, :]}")
    #
    # Ps[5, :, :] = P6
    # print(f"\nCAMERA 6:")
    # print(f"\t{P6[0, :]}")
    # print(f"\t{P6[1, :]}")
    # print(f"\t{P6[2, :]}")
    #
    # Ps[6, :, :] = P7
    # print(f"\nCAMERA 6:")
    # print(f"\t{P7[0, :]}")
    # print(f"\t{P7[1, :]}")
    # print(f"\t{P7[2, :]}")

    print("\n" + "--" * 7 + " Stage 4: Compute Initial Reconstruction using P0, P1 " + "--" * 7)

    # these are my points...
    points3D = np.full((points[0].shape[0], 4), -1.0)

    initialStructure1 = triangulatePoints(P1, P2, [points[0], points[1]], 0, 1, visibilityMat)
    points3D[:, :] = initialStructure1

    error = simpleReprojectionError(points3D, np.array([P1, P2]), points, [0, 1], visibilityMat)

    print(f"\nComputed Total Initial Reprojection error: {error}")
    print(f"Average Initial Reprojection error: {error / points.shape[1]}")
    print(f"\nIf the above are low, then lets do P1, P2")
    #
    # points3D = np.full((points[0].shape[0], 4), -1.0)
    #
    # initialStructure1 = triangulatePoints(P1, P3, [points[0], points[2]], 0, 2, visibilityMat)
    # points3D[:, :] = initialStructure1
    #
    # error = simpleReprojectionError(points3D, np.array([P1, P3]), points, [0, 2], visibilityMat)
    #
    # print(f"\nComputed Total Initial Reprojection error: {error}")
    # print(f"Average Initial Reprojection error: {error / points.shape[1]}")
    # print(f"\nIf the above are low, then lets do P1, P3")

    self_calibrate(Ps)


if __name__ == '__main__':
    main()
