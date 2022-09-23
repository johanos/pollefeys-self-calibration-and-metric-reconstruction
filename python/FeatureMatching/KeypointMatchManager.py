from typing import Dict, List, Optional
from python.FeatureMatching.Keypoint import Keypoint
from python.FeatureMatching.Image import Image
from python.FeatureMatching.KeypointExtractor import NUMBER_OF_ORB_FEATURES
import numpy as np
import matplotlib.pyplot as plt

import cv2

# I will have n images for those n images I will be able to find k keypoints in each.
# For those k keypoints they will be present in at least 1 image and at most all n images
# each keypoint represents a 3D point
# this class is meant to keep track of this


class KeypointMatchManager(object):
    # keypoint id to -> List[Keypoint] which will be a thing we can use to track if a keypoint is present in an image
    match_sequence: Dict[int, List[Optional[Keypoint]]]
    interest_point_id_seed: int

    # easy to just make id's as increments and if I delete one or something
    # just ignore that deletion maybe use a queue for id's that can be reused

    def __init__(self):
        self.interest_point_id_seed = 0
        self.match_sequence = {}
        self.full_matches = None
        self.first_matches = []

    def compute_matches(self, image_1: Image, image_2: Image, generate_dictionary=False):
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        image_1_descriptors = image_1.get_descriptors()
        image_2_descriptors = image_2.get_descriptors()

        matches_bf = bf_matcher.match(image_1_descriptors, image_2_descriptors)
        matches_bf = sorted(matches_bf, key=lambda x: x.distance)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        des1 = np.float32(image_1_descriptors)
        des2 = np.float32(image_2_descriptors)

        matches = flann.knnMatch(des1, des2, k=2)

        # ratio test as per Lowe's paper
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)

        matches_bf = good

        if generate_dictionary:
            match_dict = {}
            for match in matches_bf:

                keypoint_in_1_id = match.queryIdx  # image_1
                keypoint_in_2_id = match.trainIdx  # image_2
                match_dict[keypoint_in_1_id] = keypoint_in_2_id

            return match_dict
        return matches_bf

    def compute_matches_between(self, image_1: Image, image_2: Image, sequence_len: int, is_first):
        matches_bf = self.compute_matches(image_1, image_2)

        for match in matches_bf:
            keypoint_in_query_id = match.queryIdx  # image_1
            keypoint_in_train_id = match.trainIdx  # image_2

            # if they're near the center ignore
            point_1_uv = image_1.get_keypoint(keypoint_in_query_id).uv_position
            point_2_uv = image_2.get_keypoint(keypoint_in_train_id).uv_position

            if 400 < point_1_uv[0] < 650 and 100 < point_1_uv[1] < 480:
                continue
            if 400 < point_2_uv[0] < 650 and 100 < point_2_uv[1] < 480:
                continue

            if is_first:
                self.first_matches.append((keypoint_in_query_id, keypoint_in_train_id))

            # if I store it as a list where each index is an image then at each index I can have an object by keypoint_id
            # I will have at most N matches and each will have an array of size S (sequence length)

            # this would try to add it to existing onces, if successful would continue or something else its a new one...
            added_to_sequence = False
            for point in self.match_sequence:
                # check if there's any matches that contain the keypoint in image 1 for that id
                # this would relate it to an existing one
                current_seq = self.match_sequence[point]
                kp_at_1: Keypoint = current_seq[image_1.id_num]
                if kp_at_1 is None:
                    # should add to sequence
                    continue

                if kp_at_1.index == keypoint_in_query_id:
                    current_seq[image_2.id_num] = image_2.get_keypoint(keypoint_in_train_id)
                    added_to_sequence = True
                    break

            if added_to_sequence:
                continue

            # if I get here these are new matches
            interest_point_id = self.generate_interest_point_id()
            result: List[Optional[Keypoint]] = [None] * sequence_len
            result[image_1.id_num] = image_1.get_keypoint(keypoint_in_query_id)
            result[image_2.id_num] = image_2.get_keypoint(keypoint_in_train_id)

            self.match_sequence[interest_point_id] = result

        # Print total number of matching points between the training and query images
        print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches_bf))

    def find_only_full_matches(self):
        full_count = 0
        interest_points_full = []
        for interest_point_id in self.match_sequence:

            keypoints_for_interest_point = self.match_sequence[interest_point_id]
            is_full = True

            for kp in keypoints_for_interest_point:
                if kp is None:
                    is_full = False
                    break

            if is_full:
                interest_points_full.append(interest_point_id)
                full_count += 1

        self.full_matches = interest_points_full
        return full_count, self.full_matches

    def generate_interest_point_id(self):
        """
        :return: changes keypoint_id_seed and returns what it was before to use as a keypoint_id
        """
        new_id = self.interest_point_id_seed
        self.interest_point_id_seed += 1
        return new_id

    def visibility_matrix(self, only_full_matches) -> np.ndarray:
        # use self.matches to generate a visibility matrix...
        # this will be useful in many places.

        vizMat: np.ndarray

        if only_full_matches:
            # store a mapping from index -> interest point
            num_points = len(self.full_matches)
            num_images = len(self.match_sequence[self.full_matches[0]])
            index_to_entry = {}
            vizMat = np.full((num_images, num_points), False)

            for i, interest_point_id in enumerate(self.full_matches):
                index_to_entry[i] = interest_point_id
                seq = self.match_sequence[interest_point_id]

                for j, kp in enumerate(seq):
                    if kp is not None:
                        vizMat[j, i] = True

        else:
            num_points = len(self.match_sequence)
            num_images = len(self.match_sequence[0])
            index_to_entry = {}
            vizMat = np.full((num_images, num_points), False)

            for i, interest_point_id in enumerate(self.match_sequence):
                index_to_entry[i] = interest_point_id
                keypoints_for_interest_point = self.match_sequence[interest_point_id]
                for j, kp in enumerate(keypoints_for_interest_point):
                    if kp is not None:
                        # because my data is going to be in "num_images" x "num_points" x "3"
                        vizMat[j, i] = True

        return vizMat

    def measurement_matrix(self, only_full_matches):

        if not only_full_matches:
            raise NotImplemented

        num_points = len(self.full_matches)
        num_images = len(self.match_sequence[self.full_matches[0]])
        index_to_entry = {}
        # uv homogeneous coordinates
        measurement_matrix = np.ones((num_images, num_points, 3))

        for i, interest_point_id in enumerate(self.full_matches):
            index_to_entry[i] = interest_point_id
            seq = self.match_sequence[interest_point_id]

            for j, kp in enumerate(seq):
                if kp is not None:
                    uv_coordinates = np.ones(3)
                    uv_coordinates[0] = kp.uv_position[0]
                    uv_coordinates[1] = kp.uv_position[1]

                    measurement_matrix[j, i, :] = uv_coordinates

        return measurement_matrix
