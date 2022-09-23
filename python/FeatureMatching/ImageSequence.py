from typing import List, Optional
from python.FeatureMatching.Image import Image
from python.FeatureMatching.KeypointMatchManager import KeypointMatchManager
from python.FeatureMatching.KeypointExtractor import KeypointExtractor
from python.FeatureMatching.Keypoint import Keypoint

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import DMatch
import cv2


class ImageSequence(object):
    images: List[Image]  #it is an Image List
    keypoint_match_manager: KeypointMatchManager  # will manage the detection and generation of keypoints
    keypoint_extractor: KeypointExtractor
    image_id_seed: int
    full_matches: List[List[int]]

    def __init__(self):
        self.keypoint_match_manager = KeypointMatchManager()
        self.keypoint_extractor = KeypointExtractor()
        self.image_id_seed = 0
        self.images = []

    def add_image_with_uri(self, im_uri: str):
        # for each incoming image,
        # 1. Call the Keypoint Extractor instance function to look at an image and generate the keypoints
        # 2. Update the facilities for managing them
        im_id: int = self.generate_image_id()
        im = Image(im_uri, im_id)
        # this will extract the keypoints from each image as well as the descriptors.
        self.keypoint_extractor.extract_keypoints_from_image(im)
        self.images.append(im)

    def show_sequence(self) -> None:
        # this will make a plt plot like I did for the images
        num_images_in_sequence = len(self.images)
        fig, plots = plt.subplots(math.ceil(num_images_in_sequence / 2), 2, figsize=(20, 10))

        for i, im in enumerate(self.images):
            plots[i // 2][i % 2].set_title(f"file {im.title}, id: {im.id_num}")
            plots[i // 2][i % 2].imshow(im.load_image())
        plt.show()

    def generate_matches_for_sequence(self, do_full_matches):
        # cheap way would be to just get each pair of images

        for i in range(len(self.images) - 1):
            image_1 = self.images[i]
            image_2 = self.images[i + 1]

            self.keypoint_match_manager.compute_matches_between(image_1, image_2, len(self.images), i == 0)

        if do_full_matches:
            full_count, full_matches = self.keypoint_match_manager.find_only_full_matches()
            print(f"Number of keypoints found in all images {full_count}")
        else:
            full_matches = self.keypoint_match_manager.match_sequence
        return full_matches

    def show_correspondences(self, full_matches: List[Optional[int]]):
        for i in range(len(self.images) - 1):
            image_1_id = i
            image_2_id = i + 1
            self.show_correspondences_between(full_matches, image_1_id, image_2_id)

    def show_correspondences_between(self,
                                     full_matches: List[Optional[int]],
                                     image_id_1: int = 0,
                                     image_id_2: int = 1):
        image_1: Image = self.images[image_id_1]
        image_2: Image = self.images[image_id_2]

        image_1_keypoints = image_1.get_raw_keypoints()
        image_2_keypoints = image_2.get_raw_keypoints()

        matches = []
        for interest_point in full_matches:
            seq = self.keypoint_match_manager.match_sequence[interest_point]
            kp_1 = seq[image_1.id_num]
            kp_2 = seq[image_2.id_num]

            match = DMatch(kp_1.index, kp_2.index, 1)
            matches.append(match)

        result = cv2.drawMatches(
            image_1.load_image(),
            image_1_keypoints,
            image_2.load_image(),
            image_2_keypoints,
            matches,
            None,
            flags=2)

        plt.imshow(result)
        plt.title(f"Correspondences between image {image_1.id_num} and {image_2.id_num}")
        plt.show()

    def generate_image_id(self):
        """
        :return: changes keypoint_id_seed and returns what it was before to use as a keypoint_id
        """
        new_id: int = self.image_id_seed
        self.image_id_seed += 1
        return new_id

    def generate_visibility_matrix(self, only_full_points=True):
        return self.keypoint_match_manager.visibility_matrix(only_full_points)

    def generate_measurement_matrix(self, only_full_points=True):
        return self.keypoint_match_manager.measurement_matrix(only_full_points)

    def show_first_matches(self):

        image_1: Image = self.images[0]
        image_2: Image = self.images[1]

        image_1_keypoints = image_1.get_raw_keypoints()
        image_2_keypoints = image_2.get_raw_keypoints()
        matches = []
        for interest_point in self.keypoint_match_manager.first_matches:
            kp_1_id = interest_point[0]
            kp_2_id = interest_point[1]

            match = DMatch(kp_1_id, kp_2_id, 1)
            matches.append(match)

        result = cv2.drawMatches(
            image_1.load_image(),
            image_1_keypoints,
            image_2.load_image(),
            image_2_keypoints,
            matches,
            None,
            flags=2)

        plt.imshow(result)
        plt.title(f"First correspondences between image {image_1.id_num} and {image_2.id_num}")
        plt.show()
