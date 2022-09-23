import numpy as np
import matplotlib.pyplot as plt
from typing import List
from python.FeatureMatching.Keypoint import Keypoint
import os
import cv2


class Image(object):
    uri = "/default.png"
    id_num = -1
    im_ref: np.ndarray = None
    keypoints: List[Keypoint]
    title: str = ""

    def __init__(self, uri: str, id_num: int):
        self.id_num = id_num
        self.uri = uri
        self.title = os.path.basename(uri)
        self.keypoints = []

    def load_image(self) -> np.ndarray:
        if self.im_ref is None:
            self.im_ref = plt.imread(self.uri)
            # This is to normalize between 0 - 255
            self.im_ref = cv2.normalize(
                src=self.im_ref, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if self.im_ref.shape[2] > 3:
                self.im_ref = self.im_ref[:, :, :3]
        return self.im_ref

    def register_keypoint(self, keypoint: Keypoint) -> bool:
        old_len = len(self.keypoints)

        self.keypoints.append(keypoint)

        if len(self.keypoints) == (old_len + 1):
            return True
        return False

    def get_keypoint(self, i):
        return self.keypoints[i]

    def get_descriptors(self):
        if len(self.keypoints) == 0:
            raise ValueError(f"keypoints array is empty for image with id: {self.id_num}")
        # Todo: store this better
        descriptors = []
        for keypoint in self.keypoints:
            descriptors.append(keypoint.descriptor)

        return np.array(descriptors)

    def get_raw_keypoints(self):
        if len(self.keypoints) == 0:
            raise ValueError(f"keypoints array is empty for image with id: {self.id_num}")

        keypoints_raw = []
        for keypoint in self.keypoints:
            keypoints_raw.append(keypoint.data)

        return np.array(keypoints_raw)
