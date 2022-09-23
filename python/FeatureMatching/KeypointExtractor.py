# TBD for ORB features and their mathcing
# from https://gist.github.com/deepanshut041/d73d20bfd77eb27c96090d2a3c171fcc#file-orb-ipynb
# based on http://www.willowgarage.com/sites/default/files/orb_final.pdf
from python.FeatureMatching.Image import Image
from python.FeatureMatching.Keypoint import Keypoint
import cv2
import matplotlib.pyplot as plt
import numpy as np
NUMBER_OF_ORB_FEATURES = 600


class KeypointExtractor(object):

    def __init__(self):
        # whatever instance variables I need and whatever
        None

    def extract_keypoints_from_image(self, image: Image):
        orb_detector = cv2.ORB_create(nfeatures=NUMBER_OF_ORB_FEATURES)
        print(image)

        raw_image = image.load_image()

        # raw_image = cv2.GaussianBlur(raw_image, (5, 5),
        #                              0)  # blur to remove high frequency artifacts/interference

        gray_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
        image_keypoints, image_descriptors = orb_detector.detectAndCompute(gray_image, None)
        # Todo: some sort of facility for color designation
        white = 0xffffff

        for index, info in enumerate(zip(image_keypoints, image_descriptors)):
            keypoint, descriptor = info
            keypoint_object = Keypoint(white, keypoint.pt, keypoint, descriptor, image.id_num, index)
            image.register_keypoint(keypoint_object)

        print(f"Number of Keypoints Detected in {image.title}:", len(image_keypoints))

        return image.keypoints
