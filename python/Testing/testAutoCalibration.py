from python.AutoCalibration.autoCalibration import *
from python.AutoCalibration.DLT import factor
from random import random
from scipy.spatial.transform import Rotation as R
import math
np.set_printoptions(suppress=True)


def test_KFromAbsoluteConic():
    print("Testing getting K from image of absolute conic")
    K = np.array([[10, 1, 30], [0, 20, 40], [0, 0, 1]], dtype=np.float32)

    w = np.linalg.inv(K @ K.T)

    Kp = KfromAbsoluteConic(w)

    assert (np.all(K - Kp) < 1E-6)

    print(f"Given\n{K}")
    print(f"Reconstructed\n{Kp}")


def test_KFromAbsoluteConicSignedDiagonal():
    print("Testing getting K from image of absolute conic negative diagonal")
    K = np.array([[10, 1, 30], [0, -20, 40], [0, 0, -1]], dtype=np.float32)

    # columns changed to make diagonal positive...
    Kpositive = np.array([[10, 1, -30], [0, 20, -40], [0, 0, 1]], dtype=np.float32)

    w = np.linalg.inv(K @ K.T)
    Kp = KfromAbsoluteConic(w)

    assert (np.all(Kpositive - Kp) < 1E-8)

    print(f"Given\n{K}")
    print(f"Reconstructed\n{Kp}")


def test_MetricTransformationMetricInput():
    width = 1000
    height = 1000

    # 1000x800 image with 35mm equiv focal length.
    K = np.array([[width, 0, width / 2 - 1], [0, width, height / 2 - 1], [0, 0, 1]], dtype=np.float32)

    a = AutoCalibrationLinear()

    # add cameras with random rotation and translation

    for i in range(4):
        # zyx
        xRot = R.from_euler('x', random() * 2 * math.pi, degrees=False)
        yRot = R.from_euler('y', random() * 2 * math.pi, degrees=False)
        zRot = R.from_euler('z', random() * 2 * math.pi, degrees=False)

        Rotation = xRot * yRot * zRot

        t = np.array([random() * 10, random() * 10, random() * 10])

        M = np.empty((3, 4))
        M[:3, :3] = Rotation.as_dcm()
        M[:3, 3] = t

        P = K @ M
        a.addProjection(P, width, height)

    H, DAQ = a.metricTransformation()

    Rotation = H[:3, :3]
    RRt = Rotation @ Rotation.T
    id = np.identity(3)

    assert (expectMatrixProp(id, RRt, 1E-2))

    p = H[3, :]

    #
    print(f"RRT is\n{RRt/ RRt[-1,-1]}")
    print(f"plane at infinity is:\n{p}")


def test_RandomInputMetricCalibration():

    numCams = 5
    # width = 1000
    # height = 800
    width = 1000
    height = 1000

    # 1000x800 image with 35mm equiv focal length.
    K = np.array([[width, 0, (width - 1) / 2 - 1], [0, height, (height - 1) / 2 - 1], [0, 0, 1]],
                 dtype=np.float32)

    hReal = np.array([[1, 0, -4, 2], [1, -1, 1, -7], [-4, 0, 2, 0], [1, 2, 3, 1]], dtype=np.float32)

    a = AutoCalibrationLinear()
    Ps = np.zeros((numCams, 3, 4))
    for i in range(numCams):
        # zyx
        xRot = R.from_euler('x', random() * 2 * math.pi, degrees=False)
        yRot = R.from_euler('y', random() * 2 * math.pi, degrees=False)
        zRot = R.from_euler('z', random() * 2 * math.pi, degrees=False)

        Rotation = xRot * yRot * zRot

        t = np.array([random() * 100, random() * 100, random() * 100])

        M = np.empty((3, 4))
        M[:3, :3] = Rotation.as_dcm()
        M[:3, 3] = t

        Pmetric = K @ M
        Ps[i] = Pmetric @ np.linalg.inv(hReal)  # distort

        a.addProjection(Ps[i], width, height)

    hComputed, DAQ = a.metricTransformation()

    print(f"K in:\n{K}")
    for i in range(numCams):
        Pmetric = Ps[i] @ hComputed  # undistort the cams
        Kcomputed, Rotation, t = factor(Pmetric)
        print(f"K out:\n{Kcomputed}")


def expectMatrixProp(a, b, tolerance):
    dimsMatch = a.shape == b.shape

    if dimsMatch:
        c = cosinus(a, b)
        if c * c < 1:
            s = np.sqrt(1 - c * c)
            assert (s - tolerance < 1E-2)
            return True
    return False


def cosinus(a, b):
    return (a.ravel() @ b.ravel()) / (np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro"))


if __name__ == '__main__':
    print(f"==== TEST 1 ====")
    test_KFromAbsoluteConic()
    print(f"==== TEST 2 ====")
    test_KFromAbsoluteConicSignedDiagonal()
    print(f"==== TEST 3 ====")
    test_MetricTransformationMetricInput()
    print(f"==== TEST 4 ====")
    test_RandomInputMetricCalibration()
