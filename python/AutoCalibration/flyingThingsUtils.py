import numpy as np
import cv2
from collections import namedtuple
import math


def clamp(value, maxVal, minVal):
    return max(min(value, maxVal), minVal)


def draw_points_in_image(positions, im, width=3):
    for pos in positions:
        draw_square_at_pos(pos, im, width)


def draw_square_at_pos(pos, im, width):
    # POS is X,Y in images x coordinates are the column (second dim)
    # y coordinates are the row (first dim)
    x = pos[0]
    y = pos[1]
    if math.isnan(x) or math.isnan(y):
        return
    x = int(x)
    y = int(y)
    if x < 0 or x >= im.shape[1]:
        return
    if y < 0 or y >= im.shape[0]:
        return

    radius = width
    color = (0, 255, 0)
    center = (x, y)
    thickness = 1

    im = cv2.circle(im, center, radius, color, thickness)


def opticalFlowAt(pos, im):
    if np.any(np.isnan(pos)):
        return
    x = int(pos[0])
    y = int(pos[1])
    if x < 0 or x >= im.shape[1]:
        return
    if y < 0 or y >= im.shape[0]:
        return

    return im[y, x]


def computeNewPositions(arr, ofi):
    results = np.full(arr.shape, float('nan'))
    for i, pos in enumerate(arr):
        flowVec = opticalFlowAt(pos, ofi)
        if flowVec is not None:
            newPosition = np.array([1.0, 1.0, 1.0])
            newPosition[0] = pos[0] + flowVec[0]
            newPosition[1] = pos[1] + flowVec[1]
            results[i] = newPosition

    return results


def getDataFromTxtFile(filename, use_subset=False):
    with open(filename) as f:
        lines = f.read().splitlines()
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    for i in range(number_pts):
        split_arr = lines[i + 1].split()
        if len(split_arr) == 2:
            y, x = split_arr
        else:
            x, y, z = split_arr
            points[i, 2] = z
        points[i, 0] = x
        points[i, 1] = y
    return points


def create_array_of_same_size_cleaned(pointsArray, viewIndex, visibilityMat):
    """
    :param visibilityMat: Visibility Matrix
    :param viewIndex: Index of View
    :param pointsArray: incoming array of points size N, where N is the absolute total matches computed.
    assuming no new matches...
    :return result: an array of the same size but any entry in the original one is set to all negative ones...
    """
    result = np.full(pointsArray.shape, -1.0)
    for ri in range(pointsArray.shape[0]):
        if not visibilityMat[viewIndex, ri]:
            continue

        row = pointsArray[ri, :]
        result[ri, :] = row
    return result
