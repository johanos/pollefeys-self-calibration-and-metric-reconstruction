import numpy as np
import os
import matplotlib.pyplot as plt

from python.Middlebury.readFlowFile import read as readFlow
from python.AutoCalibration.flyingThingsUtils import getDataFromTxtFile, computeNewPositions, draw_points_in_image
from python.AutoCalibration.SintelUtils import cam_read

frame_root_path = 'TestingData/Sintel'
optical_flow_root = 'TestingData/optical_flow/Sintel'
initial_points_template = "TestingData/Sintel/{}/pt_2D.txt"
opticalFlowTemplate = "frame_{}.flo"
frameImageTemplate = "frame_{}.png"

camera_params_root = 'TestingData/Sintel'

camera_params_file_template = "frame_{}.cam"


def generatePointTrajectories(sequenceName, numFrames, startFrame, generateImages=True):
    pointFile = initial_points_template.format(sequenceName)
    points = np.array([(getDataFromTxtFile(pointFile))], dtype=np.float64)

    print("---" * 40)
    print("\tComputing Points For Sequence")
    images = []
    gt_camera_matrices = []
    for next_view in range(numFrames - 1):
        suffix = str(startFrame + next_view).zfill(4)

        camera_params_file = camera_params_file_template.format(suffix)

        camera_params_file = os.path.join(camera_params_root,
                                          os.path.join(sequenceName + "_cam", camera_params_file))

        optical_flow_path = os.path.join(optical_flow_root,
                                         os.path.join(sequenceName, opticalFlowTemplate.format(suffix)))

        print(f"\tReading Optical Flow File {optical_flow_path}")

        cam_gt_params = cam_read(camera_params_file)
        gt_camera_matrices.append(cam_gt_params[0])
        opFlow = readFlow(optical_flow_path)
        temp = computeNewPositions(points[next_view], opFlow)
        print(temp[0:4])
        points = np.append(points, [temp], axis=0)

    suffix = str(startFrame + numFrames).zfill(4)
    camera_params_file = camera_params_file_template.format(suffix)
    camera_params_file = os.path.join(camera_params_root,
                                      os.path.join(sequenceName + "_cam", camera_params_file))
    final_gt_param = cam_read(camera_params_file)
    gt_camera_matrices.append(final_gt_param[0])

    if not generateImages:
        return points, gt_camera_matrices

    for view in range(numFrames):
        suffix = str(startFrame + view).zfill(4)
        frame_path = os.path.join(frame_root_path,
                                  os.path.join(sequenceName, frameImageTemplate.format(suffix)))

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

    return points, images, gt_camera_matrices


def main():
    numFrames = 5
    numPoints = 10
    startFrame = 10

    points, images = generatePointTrajectories("temple_2", 5, startFrame, True)


if __name__ == '__main__':
    main()
