from typing import Tuple


class Keypoint:
    color: int = 0xFFFFFF
    uv_position: Tuple[float, float] = (-1.0, -1.0)
    descriptor = None  # from the orb output
    data = None  # from the orb output
    residing_image_id = -1
    index = -1

    def __init__(self, color, uv_position, data, descriptor, residing_image_id, index):
        if color:
            self.color = color
        if uv_position:
            self.uv_position = uv_position
        self.data = data
        self.descriptor = descriptor
        self.residing_image_id = residing_image_id
        self.index = index

    def __hash__(self):
        return hash((self.residing_image_id, self.index))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        is_for_same_image = self.residing_image_id == other.residing_image_id
        is_same_keypoint_id = self.index == other.index
        return is_for_same_image and is_same_keypoint_id
