from python.FeatureMatching.ImageSequence import ImageSequence

from python.AutoCalibration.AutoCalibrationManager import AutoCalibrationManager

import numpy as np
import os
import glob

# good sequence is "./TestingData/SceneNetRGBD/0/3/photo/" at 180
# base_file_dir = "./TestingData/SceneNetRGBD/0/3/photo/"
# start_offset = 180

#base_file_dir = "./TestingData/frames/TEST/C/0144/left/"
base_file_dir = "./TestingData/Sintel/alley_2/"
start_offset = 20

img_sequence = ImageSequence()

images = sorted(
    glob.glob(base_file_dir + "*.png") + glob.glob(base_file_dir + "*.jpg"),
    key=lambda k: int((os.path.splitext(os.path.split(k)[1])[0])[6:]))

num_images = 4
skip = 2
for i in range(num_images):
    #image_file = os.path.join(base_file_dir, f"{start_offset+ im_count * 25}.jpg")
    im_name = images[start_offset + i * skip]
    print(im_name)
    img_sequence.add_image_with_uri(im_name)

print("\n" + "--" * 8 + " Stage 0: Load in an image sequence " + "--" * 8)

img_sequence.show_sequence()

print("\n" + "--" * 8 + " Stage 1: Point Correspondences " + "--" * 8)

full_matches = img_sequence.generate_matches_for_sequence(do_full_matches=True)

img_sequence.show_correspondences(full_matches)

img_sequence.show_first_matches()

print("\n" + "--" * 8 + " Stage 2: Compute Fundamental Matrix between view 0 and view 1" + "--" * 8)
visibility_matrix = img_sequence.generate_visibility_matrix()
measurement_matrix = img_sequence.generate_measurement_matrix()

ac_manager = AutoCalibrationManager(measurement_matrix, visibility_matrix, img_sequence)

ac_manager.compute_fundamental_matrix_between_first_views()

print("\n" + "--" * 8 + " Stage 3: Compute Initial Projection Matrices Using F12 " + "--" * 8)
ac_manager.initialize_camera_structure()

print("\n" + "--" * 7 + " Stage 4: Compute Initial Reconstruction using P0, P1 " + "--" * 7)
ac_manager.triangulate_points()
print("\n" + "--" * 6 + " Stage 5: Resection New View using Initial Reconstruction Frame " + "--" * 6)
ac_manager.resection_remaining_cameras()
print("\n" + "--" * 6 + " Stage 6: Self Calibration Begins " + "--" * 6)
ac_manager.self_calibrate()
