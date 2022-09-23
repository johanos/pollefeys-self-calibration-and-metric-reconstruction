import os
import matplotlib.pyplot as plt
import numpy as np
from python.AutoCalibration.flyingThingsUtils import getDataFromTxtFile, draw_points_in_image
import matplotlib
import math
frame_root_path = 'TestingData/SceneNetRGBD/'
optical_flow_root = 'TestingData/optical_flow/SceneNetRGBD'
from random import uniform


def getNextPositions(opticalFlow, points):
    newPoints = np.full(points.shape, float('nan'))
    for pi, point in enumerate(points):
        flowVec = getNextPosition(opticalFlow, point)
        print(flowVec)
        if flowVec is not None:
            newPosition = np.array([1.0, 1.0, 1.0])
            newPosition[0] = point[0] + flowVec[1]
            newPosition[1] = point[1] + flowVec[0]
            newPoints[pi, :] = newPosition

    return newPoints


def getNextPosition(opticalFlow, point):
    if np.any(np.isnan(point)):
        return
    x = int(point[0])
    y = int(point[1])
    if x < 0 or x >= opticalFlow.shape[1]:
        return
    if y < 0 or y >= opticalFlow.shape[0]:
        return

    # optical flow is defined as an angle (divided by 2pi)
    # and a magnitude
    # direction is in the hue channel
    # saturation is 1
    # magnitude is in the value channel ( mag * 1.0/100.0) so * 100 = mag.
    theta = opticalFlow[y, x][0] * (2.0 * math.pi)
    magnitude = opticalFlow[y, x][2] * 100.0

    xComponent = magnitude * math.cos(theta)
    yComponent = magnitude * math.sin(theta)

    return [yComponent, xComponent]


def main():
    numFrames = 3
    sequenceName = "0/223"

    opticalFlowTemplate = "optical_flow_{}.png"

    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = ((numFrames * 2) // 2) + 1
    count = 1
    images = []
    opticalFlows = []
    for i in range(numFrames):
        frameNum = i * 25
        frameDir = os.path.join(frame_root_path, os.path.join(sequenceName, "photo"))

        opticalFlowPath = opticalFlowTemplate.format(frameNum)

        frameFile = os.path.join(frameDir, f"{frameNum}.jpg")
        opticalFlowFile = os.path.join(optical_flow_root, os.path.join(sequenceName, opticalFlowPath))
        print(frameFile)
        print(opticalFlowFile)

        frame = plt.imread(frameFile)
        images.append(frame)
        opticalFlow = plt.imread(opticalFlowFile)
        opFlowHSV = matplotlib.colors.rgb_to_hsv(opticalFlow)

        opticalFlows.append(opFlowHSV)

        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(f"Frame {frameNum}")
        plt.imshow(frame)

        count += 1

        ax = fig.add_subplot(rows, columns, count)
        ax.set_title(f"Optical Flow {frameNum}")
        plt.imshow(opticalFlow)
        count += 1
    plt.show()

    pointTemplate = "TestingData/optical_flow/SceneNetRGBD/{}/pt_2D.txt"
    points1txt = pointTemplate.format(sequenceName)

    points = np.array([(getDataFromTxtFile(points1txt))], dtype=np.float64)

    print(points)

    images = np.array(images)
    for i in range(numFrames - 1):
        opFlo = opticalFlows[i]
        points2D = points[i]
        nextP = getNextPositions(opFlo, points2D)
        points = np.append(points, [nextP], axis=0)

    for i, im in enumerate(images):
        points2D = points[i]
        draw_points_in_image(points2D, im, 6)

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


if __name__ == '__main__':
    main()
