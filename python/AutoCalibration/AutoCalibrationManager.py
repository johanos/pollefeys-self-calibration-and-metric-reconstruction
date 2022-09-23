from typing import List, Optional
from python.FeatureMatching.Image import Image
from python.AutoCalibration.Camera import Camera
from python.FeatureMatching.ImageSequence import ImageSequence
from python.AutoCalibration.calibrationAPI import self_calibrate
from python.AutoCalibration.ProjectiveReconstructionHelpers import *
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import DMatch
import cv2


class AutoCalibrationManager(object):
    # This is for type hints in python
    views: List[Camera]
    observations: np.ndarray
    visibility_matrix: np.ndarray
    image_sequence: ImageSequence
    fundamental_0_1: Optional[np.ndarray]
    initial_structure: Optional[np.ndarray]
    inlier_keypoint_matches: Optional[np.ndarray]
    sequence_points_uv: Optional[np.ndarray]
    inlier_masks: List[List[int]]

    def __init__(self, measurement_correspondences, visibility_matrix, image_sequence):
        self.inlier_keypoint_matches = None
        self.cameras = []
        self.observations = measurement_correspondences
        self.visibility_matrix = visibility_matrix
        self.image_sequence = image_sequence
        self.fundamental_0_1 = None
        self.initial_structure = None
        self.inlier_keypoint_matches = None
        self.sequence_points_uv = None
        self.inlier_masks = []
        self.kp_to_i = {}

    def compute_fundamental_matrix_between_first_views(self):

        points_0, points_1 = [], []

        matching_keypoints = [
            [0, 0] for _ in range(len(self.image_sequence.keypoint_match_manager.first_matches))
        ]

        for i, match in enumerate(self.image_sequence.keypoint_match_manager.first_matches):
            kp_0_id = match[0]
            kp_1_id = match[1]

            kp_0 = self.image_sequence.images[0].get_keypoint(kp_0_id)
            kp_1 = self.image_sequence.images[1].get_keypoint(kp_1_id)

            pos_0 = np.array([kp_0.uv_position[0], kp_0.uv_position[1], 1])
            pos_1 = np.array([kp_1.uv_position[0], kp_1.uv_position[1], 1])
            points_0.append(pos_0)
            points_1.append(pos_1)
            matching_keypoints[i][0] = kp_0_id
            matching_keypoints[i][1] = kp_1_id

        points_0 = np.array(points_0)
        points_1 = np.array(points_1)
        matching_keypoints = np.array(matching_keypoints)

        fundamental_0_1, points_mask = compute_fundamental_matrix_between(points_0, points_1)
        frame_0: Image = self.image_sequence.images[0]
        frame_1: Image = self.image_sequence.images[1]

        # get inliers to show
        points_0 = filter_points_using_mask(points_0, points_mask)
        points_1 = filter_points_using_mask(points_1, points_mask)
        inliers_keypoints = filter_points_using_mask(matching_keypoints, points_mask)

        for i, inliers_keypoint in enumerate(inliers_keypoints[:, 1]):
            self.kp_to_i[inliers_keypoint] = i

        im_l, im_r = draw_epilines_and_points(frame_0.load_image().copy(),
                                              frame_1.load_image().copy(), fundamental_0_1, points_0,
                                              points_1)

        fx, plots = plt.subplots(1, 2, figsize=(20, 10))
        plots[0].set_title("Epipolar Lines in Image 0")
        plots[0].imshow(im_l)
        plots[1].set_title("Epipolar Lines in Image 1")
        plots[1].imshow(im_r)
        plt.show()

        # At this point I have a fundamental matrix calculated for these points
        # I also have a list of inliers which is a subset of the measurement_correspondences member var that I can
        print(
            f"Calculated Fundamental Matrix Inliers {np.count_nonzero(points_mask)} out of {len(points_mask)}"
        )

        self.fundamental_0_1 = fundamental_0_1
        self.inlier_keypoint_matches = inliers_keypoints
        self.sequence_points_uv = np.array([points_0, points_1])

    def initialize_camera_structure(self):
        if self.fundamental_0_1 is None:
            raise Exception("Need to compute fundamental matrix between first few views")

        # because F is computed using outlier friendly RANSAC this is okay...
        P0, P1 = create_projectively_ambiguous_camera_matrices(self.fundamental_0_1)

        camera_0 = Camera(0, None, None, None, P=P0)
        print(f"\nCAMERA 0:")
        print(f"\t{P0[0, :]}")
        print(f"\t{P0[1, :]}")
        print(f"\t{P0[2, :]}")

        camera_1 = Camera(1, None, None, None, P=P1)
        print(f"\nCAMERA 1:")
        print(f"\t{P1[0, :]}")
        print(f"\t{P1[1, :]}")
        print(f"\t{P1[2, :]}")

        self.cameras.append(camera_0)
        self.cameras.append(camera_1)

    def triangulate_points(self):

        # previously computed these guys and filtered by inliers
        points_0 = self.sequence_points_uv[0]
        points_1 = self.sequence_points_uv[1]

        P0: Camera = self.cameras[0]
        P1: Camera = self.cameras[1]

        correlated_correspondences = np.array([points_0, points_1])
        self.initial_structure = triangulate_points_no_visibility_matrix(P0.P, P1.P,
                                                                         correlated_correspondences)

        cams_to_use = [P0, P1]
        points_to_use = np.array([points_0, points_1])
        visibility_matrix = (self.inlier_keypoint_matches != -1).T
        # num_points should be something like len(cameras) * len(initial_structure)
        error_sum, num_points_processed = self.calculate_reprojection_error(self.initial_structure,
                                                                            cams_to_use, points_to_use,
                                                                            visibility_matrix)
        print(f"\tCumulative Reprojection error: {error_sum:.4f}")
        print(f"\tAverage Reprojection error: { (error_sum / num_points_processed):.4f}")
        print(
            f"\tIf cumulative and average error is low then we have initialized a reasonable projective frame"
        )
        print(f"\tIdeally both will be low")

    def resection_remaining_cameras(self):
        if self.initial_structure is None:
            raise Exception("Need to triangulate points first")

        # # at this point at view N, I should have access to N 3D points and the projection of those points in view N
        num_views = len(self.image_sequence.images)
        match_manager = self.image_sequence.keypoint_match_manager
        #visibility_matrix = (self.inlier_keypoint_matches != -1).T
        image_1 = self.image_sequence.images[1]

        len_sequences = self.sequence_points_uv.shape[1]
        points_uv = [self.sequence_points_uv[0], self.sequence_points_uv[1]]
        self.inlier_masks.append(np.ones(len_sequences))
        self.inlier_masks.append(np.ones(len_sequences))

        for i in range(2, num_views):
            # these are the points I can use after filtering out the outliers from the first 2 views
            image_at_i = self.image_sequence.images[i]
            # this is NOT filtered by the inliers for F that I used...
            match_kp1_kpi = match_manager.compute_matches(image_1, image_at_i, generate_dictionary=True)
            points_i_uv = [[-1, -1, -1] for _ in range(self.sequence_points_uv.shape[1])]

            inlier_matches = [-1] * self.sequence_points_uv.shape[1]
            for kp_entry_i, kp_in_1 in enumerate(self.inlier_keypoint_matches[:, 1]):
                if kp_in_1 in match_kp1_kpi:

                    kp_to_get = match_kp1_kpi[kp_in_1]
                    kp_obj = image_at_i.get_keypoint(kp_to_get)
                    indexer = self.kp_to_i[kp_in_1]
                    points_i_uv[indexer] = np.array([kp_obj.uv_position[0], kp_obj.uv_position[1], 1])
                    inlier_matches[indexer] = match_kp1_kpi[kp_in_1]

            inlier_matches = np.array(inlier_matches)
            inlier_visibility = (inlier_matches != -1).T
            points_i_uv = np.array(points_i_uv)

            points_uv.append(points_i_uv)
            # wittle down

            a_to_i = {}
            count = 0
            for i, inlier_flag in enumerate(inlier_visibility):
                if inlier_flag:
                    a_to_i[count] = i
                    count += 1

            points_i_uv = points_i_uv[inlier_visibility]
            points_i_3D = self.initial_structure[inlier_visibility]

            Pi, inlier_mask_view_i = self.resection_cameras_RANSAC(points_i_uv, points_i_3D)
            camera_i = Camera(i, None, None, None, P=Pi)

            inlier_mask_for_view = [0] * len_sequences

            for a, inlier_status in enumerate(inlier_mask_view_i):
                if inlier_status:
                    i = a_to_i[a]
                    inlier_mask_for_view[i] = 1

            self.inlier_masks.append(inlier_mask_for_view)

            self.cameras.append(camera_i)

        print(f"\t{num_views-2} cameras recovered")
        # now I should see the reprojection error

        points_uv = np.array(points_uv)

        filtered_visibility_matrix = np.array(self.inlier_masks)

        # right now assume I am using the same inlier points as the first 2 views, this is incorrect, but I wanna see how well it could work.
        error_sum, num_points_processed = self.calculate_reprojection_error(self.initial_structure,
                                                                            self.cameras, points_uv,
                                                                            filtered_visibility_matrix)
        print(f"\tCumulative Reprojection error: {error_sum:.4f}")
        print(f"\tAverage Reprojection error: {(error_sum / num_points_processed):.4f}")

    def get_projection_matrices_as_numpy_array(self):
        Ps = []
        # to type hint it
        cam: Camera
        for cam in self.cameras:
            Ps.append(cam.P)

        return np.array(Ps)

    def self_calibrate(self):

        im_height = self.image_sequence.images[0].load_image().shape[0]
        im_width = self.image_sequence.images[0].load_image().shape[1]
        for im in self.image_sequence.images:

            raw_im = im.load_image()

            new_im_height = raw_im.shape[0]
            new_im_width = raw_im.shape[1]
            # TODO: change this so I can pipe in arbitrary height or width (just being lazy)
            if new_im_height != im_height or new_im_width != im_width:
                raise NotImplemented("Not supporting varying width and height images yet")
            im_height = new_im_height
            im_width = new_im_width

        Ps = self.get_projection_matrices_as_numpy_array()

        Ks, isDegenerate, maxDisturbed = self_calibrate(Ps, im_width, im_height)
        print("\tComputed Intrinsic Matrices:")

        for i, K in enumerate(Ks):
            print(f"\nCAMERA {i}:")
            print(f"\t{K[0, :]}")
            print(f"\t{K[1, :]}")
            print(f"\t{K[2, :]}")

        print(f"Is this degenerate or close to degenerate?: {isDegenerate}")

    def calculate_reprojection_error(self, points_3d_collection, cameras, points_uv_collection,
                                     visibility_matrix):
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

    def resection_cameras_RANSAC(self,
                                 points_i_uv,
                                 points_i_3D,
                                 max_iters=500,
                                 inlier_thresh=5) -> (np.ndarray, np.ndarray):
        """
        :param inlier_thresh:
        :param max_iters:
        :param points_i_uv: 2D points after keeping only the points visible between the first 2 views and that are inliers. (Only ones that could match a 3D point triangulated)
        :param visibility_matrix_entry: a column vector for the view to be resectioned from the visibility matrix.
        :return:
        """
        if self.initial_structure is None:
            raise NotImplemented("Need to triangulate points first")

        # need to do RANSAC here to get a random sample
        # According to the H.Z book on page 181 this is the minimum number of samples to compute P...

        num_samples_for_fit = 10
        num_samples = points_i_uv.shape[0]

        # TODO: Test this hyperparameter
        min_inliers = 6  #int(0.1 * num_samples)

        best_camera_model = None
        best_camera_performance = 0
        best_inliers = None

        for i in range(max_iters):  # change this to a good hyper parameter at some point
            number_of_rows = points_i_uv.shape[0]
            sample_set_indices = np.random.choice(number_of_rows, size=num_samples_for_fit, replace=False)
            sample_set_uv = points_i_uv[sample_set_indices]  # should be a 6x3 tensor
            sample_set_xyz = points_i_3D[sample_set_indices]  # should be a 6x4 tensor

            camera_params_candidate = AutoCalibrationManager.fit_camera_model(sample_set_xyz, sample_set_uv)

            camera_params_performance, inliers = AutoCalibrationManager.evaluate_camera_model(
                points_i_uv, points_i_3D, camera_params_candidate, inlier_thresh)

            if camera_params_performance < min_inliers:
                continue

            if camera_params_performance > best_camera_performance:
                best_camera_model = camera_params_candidate
                best_camera_performance = camera_params_performance
                best_inliers = inliers

        print(f"Resectioned Camera with best params inlier: {best_camera_performance}")
        print(f"{(best_camera_performance / num_samples) * 100:.4f} %")
        return best_camera_model.reshape((3, 4)), best_inliers

    @classmethod
    def fit_camera_model(cls, points_3d, points_2d):
        return DLT3D_Resection(points_3d, points_2d)

    @classmethod
    def evaluate_camera_model(cls, points_2d, points_3d, camera_params, inlier_thresh):
        P = camera_params.reshape((3, 4))
        projected_points = (P @ points_3d.T)

        for i in range(3):
            projected_points[i] /= projected_points[2]

        projected_points = projected_points.T

        inlier_count = 0
        inliers = np.zeros((points_2d.shape[0]))
        count = 0
        for measured_point, projected_estimate in zip(points_2d, projected_points):
            squared_distance = np.linalg.norm(projected_estimate - measured_point)**2

            if squared_distance < inlier_thresh:
                inlier_count += 1
                inliers[count] = 1
            count += 1

        return inlier_count, inliers
