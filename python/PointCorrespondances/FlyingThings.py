import numpy as np
import os
import matplotlib.pyplot as plt

from python.AutoCalibration.readPfm import readPFM
from python.AutoCalibration.flyingThingsUtils import getDataFromTxtFile, computeNewPositions, draw_points_in_image
from python.AutoCalibration.calibrationAPI import flyingThingsIntrinsicMatrix

frame_root = "TestingData/frames/TEST/"
frame_template = "{}/left/{}.png"

optical_flow_root = "TestingData/optical_flow/TEST/"
optical_flow_template = "{}/into_future/left/OpticalFlowIntoFuture_{}_L.pfm"

pointTemplate = "TestingData/frames/TEST/{}/left/pt_2D.txt"


def generatePointTrajectories(sequence, numFrames, startFrame, generateImages=True):
    points1txt = pointTemplate.format(sequence, startFrame)
    points = np.array([(getDataFromTxtFile(points1txt))], dtype=np.float64)
    #Fs = np.zeros((numFrames, 3, 3))

    images = []
    gt_intrinsics = []
    print("---" * 40)
    print("\tComputing Points For Sequence")
    for view in range(0, numFrames - 1):
        suffix = str(startFrame + view).zfill(4)

        optical_flow_file = optical_flow_template.format(sequence, suffix)
        optical_flow_next = os.path.join(optical_flow_root, optical_flow_file)

        next_of, scale = readPFM(optical_flow_next)
        print(f"\tReading Optical Flow File {optical_flow_next}")
        temp = computeNewPositions(points[view], next_of)
        points = np.append(points, [temp], axis=0)
        gt_intrinsics.append(flyingThingsIntrinsicMatrix)

    gt_intrinsics.append(flyingThingsIntrinsicMatrix)

    if not generateImages:
        return points

    for view in range(numFrames):
        suffix = str(startFrame + view).zfill(4)
        frame_path = os.path.join(frame_root, frame_template.format(sequence, suffix))

        image = plt.imread(frame_path)
        images.append(image)

    for i, im in enumerate(images):
        points2D = points[i]
        draw_points_in_image(points2D, im, 13)

    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = (numFrames // 2) + 1
    count = 1

    for frameNum, frame in enumerate(images):
        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(f"Frame {frameNum}")
        plt.imshow(frame)
        count += 1

    plt.show()

    return points, images, gt_intrinsics
