class Observation(object):

    def __init__(self, cam_id: int, pt_id: int, x_gt, y_gt):
        self.cam_id = cam_id
        self.pt_id = pt_id
        self.x_gt = x_gt
        self.y_gt = y_gt
