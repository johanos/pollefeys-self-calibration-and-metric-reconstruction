from python.AutoCalibration.flyingThingsUtils import getDataFromTxtFile,\
    computeNewPositions
from python.AutoCalibration.readPfm import readPFM
from python.AutoCalibration.projectiveReconstruct import createProjectivelyAmbiguousCameraMatrices, \
    computeFundamentalMatrixBetween
from python.AutoCalibration.selfCalibration import centralizedPoints, selfCalibrate
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    numFrames = 4
    startFrame = 6
    frameNames = [""] * numFrames
    for frame in range(numFrames):
        frameNames[frame] = str(startFrame + frame).zfill(4)
    sequence= "C/0144"
    print("---" * 40)
    print ("\tRunning Pipeline")
    print("\n" + "==" * 24 + " Stage 1: Point Correspondences " + "==" * 27 )

    frameTemplate = "TestingData/frames/TEST/{}/left/{}.png"
    frameOFTemplate = "TestingData/optical_flow/TEST/{}/into_future/left/OpticalFlowIntoFuture_{}_L.pfm"
    pointTemplate = "TestingData/frames/TEST/{}/left/pt_2D_{}.txt"
    points1txt = pointTemplate.format(sequence, startFrame)

    points = np.array([ np.int32(getDataFromTxtFile(points1txt))])
    Fs = np.zeros((numFrames, 3,3))

    images = []
    print("---" * 40)
    print ("\tComputing Points For Sequence")
    for i in range(0, numFrames - 1):
        nextOFfile = frameOFTemplate.format(sequence, str(startFrame + i).zfill(4))
        nextOF, scale = readPFM(nextOFfile)
        print(f"\tReading Optical Flow File {nextOFfile}")
        temp = np.int32(computeNewPositions(points[i], nextOF))
        points = np.append(points, [temp], axis=0)

    frame1File = frameTemplate.format(sequence, str(startFrame).zfill(4))
    frame1 = plt.imread(frame1File)
    imHeight = frame1.shape[0]
    imWidth = frame1.shape[1]

    v2pMatrix = np.zeros((points.shape[0], points.shape[1]))

    for i in range(numFrames):
        for j,point in enumerate(points[i]):
            if all(point[0:2] >= 0) and point[0] < imWidth  and point[1] < imHeight:
                v2pMatrix[i,j] = True
            else:
                v2pMatrix[i,j] = False



    print("==" * 27 + " Stage 1 Done " + "==" * 27 )
    print("\n" + "==" * 24 + " Stage 2 Compute Fundamental Matrices Normally" + "==" * 27 )

    print("\tComputing Fundamental Matrix for First View and View N")
    print("---" * 40)
    print(f"\tReading Frame 0 {frame1File}")
    for i in range(1, numFrames):
        frameNFile = frameTemplate.format(sequence, str(startFrame + i).zfill(4))
        frameN = plt.imread(frameNFile)

        print(f"\tReading Frame {i}: {frameNFile}")
        print("\tComputing Fundamental Matrix With 8 Points Algorithm")
        pointsA, pointsB = cleanUpPointCorrespondances(points[0], points[i], v2pMatrix[0], v2pMatrix[i])
        print(f"\tPointsA and PointsB have {pointsA.shape} shape")

        F = computeFundamentalMatrixBetween(pointsA, pointsB)
        Fs[i,:,:] = F
        imL, imR = drawEpilinesAndPoints(frame1, frameN, F, points[0], points[i])
        print("---" * 40)
        images.extend([imL, imR])

    # computes the truth value of this array about axes 1 and 2 which are the 3,3 F matrices
    # will return the number of non zero element F matrices it computed.
    # it is collected like But r
    #          F1   F2   F3   F4 .... Fn
    # F1 = [                            ]  <--- TODO if i want to make it more robust


    nonEmptyFs = np.count_nonzero(np.all(Fs, axis=(1,2)))
    print (f"\tNumber of Fundamental Matrices Successfully Computed: {nonEmptyFs}")
    fig = plt.figure(figsize=(10, 10))
    columns = 2
    rows = (len(images) // 2) + 1
    count = 1
    for i in range(0, len(images), 2):
        img = images[i]
        img2 = images[i+1]
        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(f"Frame {frameNames[0]} and Frame {frameNames[i//2 + 1]}")
        plt.imshow(img)
        count += 1
        ax2 = fig.add_subplot(rows, columns, count)
        ax2.set_title(f"Frame {frameNames[0]} and Frame {frameNames[i//2 + 1]}")
        plt.imshow(img2)
        count += 1
    plt.show()

    print("\n" + "==" * 24 + " Stage 2b Compute Fundamental Matrices Transformed " + "==" * 27)
    # move all image points by the principal points...
    normalizedPoints = centralizedPoints(points)
    print("\tComputing Normalized Fundamental Matrix for First View and View N")
    print("---" * 40)
    #
    for i in range(1, numFrames):
        print(f"\tPoints for Frame {i}: {frameNames[i]}")
        print("\tComputing Fundamental Matrix With 8 Points Algorithm")
        pointsA, pointsB = cleanUpPointCorrespondances(normalizedPoints[0], normalizedPoints[i], v2pMatrix[0], v2pMatrix[i])
        print(f"\tPointsA and PointsB have {pointsA.shape} shape")
        F = computeFundamentalMatrixBetween(pointsA, pointsB)
        Fs[i, :, :] = F

    print("---" * 40)

    nonEmptyFs = np.count_nonzero(np.all(Fs, axis=(1, 2)))
    print(f"\tNumber of Fundamental Matrices Successfully Computed: {nonEmptyFs}")
    print("==" * 27 + " Stage 2 Done " + "==" * 27 )

    print("\n" + "==" * 24 + " Stage 3: Compute Initial Projection Matrices Using Fundamental Matrix " + "==" * 27 )

    # These are my camera matrices..
    Ps = np.zeros((numFrames, 3,4))

    # The very first one should be the identity
    print(f"\tComputing Camera Matrix as Identity for: Frame-{frameNames[0]}")
    Ps[0, :, :] = np.concatenate( (np.identity(3), np.zeros((1,3)).T), axis=1)

    for i in range(1, Ps.shape[0]):
        print (f"\tComputing Camera Matrix for: Frame-{frameNames[i]}")
        tempF = Fs[i]
        # This is the Fundamental Matrix between view 0 and view i
        P, Pprime = createProjectivelyAmbiguousCameraMatrices(tempF)
        Ps[i, :, :] = Pprime


    print("\n" + "==" * 24 + " Stage 3b: Compute Initial Projection Matrices Using Other Method as above is probably wrong..." + "==" * 27 )
    # These are my camera matrices..
    #Ps2 = sturmTriggsProjectiveFactorization(points, v2pMatrix)
    #assert (Ps2.shape == (numFrames, 3,4))


    print("==" * 27 + " Stage 3 Done " + "==" * 27 )
    print ("With these normalized points I have an intrinsic matrix of shape")
    print ("[ f 0 0 ]\n[ 0 f 0 ]\n[ 0 0 1 ]")
    print("\n" + "==" * 24 + " Stage 4: Self Calibration Begins " + "===" * 27)
    #points normalized.
    focalLength = selfCalibrate(Ps)
    print (f"Linear Result for Focal Lenght is: {focalLength}")
    print (f"\tDone")

    # # print("\n" + "==" * 24 + " Stage Sanity Check: Linear Triangulation using DLT with Calibrated Params " + "==" * 27)
    # # projectivePoints3D = triangulateEnMasse(points, Ps2)
    # #
    # #
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = Axes3D(fig)
    # # ax.scatter(projectivePoints3D[:, 0], projectivePoints3D[:, 1], projectivePoints3D[:, 2],
    # #            c='k', depthshade=True, s=2)
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Z')
    # # ax.set_xlim(-30, 30)
    # # ax.set_ylim(-30, 30)
    # # ax.set_zlim(-1, 20)
    # # # elevation and azimuth.
    # # ax.view_init(10, -20)
    # #
    # # plt.show()
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
