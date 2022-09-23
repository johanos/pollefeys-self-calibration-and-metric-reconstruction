from python.AutoCalibration.autoCalibration import *
from python.AutoCalibration.DLT import factor
# This is my ground TRUTH
flyingThingsIntrinsicMatrix = np.array([[1050.0, 0.0, 479.5], [0.0, 1050.0, 269.5], [0.0, 0.0, 1.0]])


# jacobian has a simple form dw(i,j) / dQ(k,m) = P[i,k] * P[j,m]
def derivative_w_daq(P, i, j, k, m, symmetric):
    if symmetric:
        dW_dQ = P[i - 1, k - 1] * P[j - 1, m - 1] + P[j - 1, m - 1] * P[i - 1, k - 1]
    else:
        dW_dQ = P[i - 1, k - 1] * P[j - 1, m - 1]
    return dW_dQ


class Constraint:

    def __init__(self, view, P, ci, i, j):
        self.constraint_index = ci
        self.view = view
        self.P = P
        self.i = i
        self.j = j

        # A quadric can be defined as   Q = a, b, c, d
        #                                   b, e, f, g
        #                                   c, f, h, i
        #                                   d, g, i, j
        #
        # a.k.a parametrized by 10 parameters

        # a  ,     b,        c,      d,       e,       f,       g,       h,      i,     j
        self.quadric_constaints_symmetry = [False, True, True, True, False, True, True, False, True, False]
        self.quadric_constaints = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4),
                                   (4, 4)]

    def generate_jacobian_row(self):
        j_row = []

        for index, q_c in enumerate(self.quadric_constaints):
            jac_entry = derivative_w_daq(self.P, self.i, self.j, q_c[0], q_c[1],
                                         self.quadric_constaints_symmetry[index])
            j_row.append(jac_entry)

        j_row = np.array(j_row)
        return j_row


class SensitivityConstaint:

    def __init__(self, num_views, first: Constraint, second=None):
        # these are the ones that should vary because in a metric case these are the ones that fully define the symmetric quadric
        # Q is symmetric (10 params) but Q_metric = diag(1,1,1,0) (-1 param) and and defined up to scale (-1 param)
        # so I only need 8 parameters to define it
        self.first = first
        self.second = second
        self.num_views = num_views
        self.num_constaints_per_view = 3

    def generate_constaint_entry(self):
        row_entry = np.zeros(self.num_constaints_per_view * self.num_views)
        # set a 1 where first dictates, set a -1 where second does (do nothing is None)

        row_entry[self.first.constraint_index] = 1
        if self.second is not None:
            row_entry[self.second.constraint_index] = -1

        return row_entry


def prepare_sensitivity(original_projection_matrices, metric_projection_matrices, DAQ, Ks):
    # assuming constant
    no_skew = True
    equal_fx_fy = True
    same_fx = True

    num_views = len(metric_projection_matrices)
    num_constraints = 3  # s = 0 fx = fy, fx = f_1x (the same throughout)
    constraintMatrix = []
    jacobian = []

    constraint_count = 0

    for view_num, Pmat in enumerate(metric_projection_matrices):
        if equal_fx_fy:
            first = Constraint(view_num, Pmat, constraint_count, 1, 1)
            jacobian.append(first.generate_jacobian_row())
            constraint_count += 1

            second = Constraint(view_num, Pmat, constraint_count, 2, 2)
            jacobian.append(second.generate_jacobian_row())
            constraint_count += 1

            equal_fx_fy_sensitivity = SensitivityConstaint(num_views, first, second)
            constraintMatrix.append(equal_fx_fy_sensitivity.generate_constaint_entry())

        if no_skew:
            first = Constraint(view_num, Pmat, constraint_count, 1, 2)
            jacobian.append(first.generate_jacobian_row())
            constraint_count += 1

            no_skew_sensitivity = SensitivityConstaint(num_views, first)
            constraintMatrix.append(no_skew_sensitivity.generate_constaint_entry())

        if same_fx:
            # no need to add here because first is already there as the 0th constaint
            if view_num + 1 >= num_views:
                continue
            first = Constraint(0, metric_projection_matrices[0], 0, 1, 1)
            second = Constraint(view_num + 1, Pmat, constraint_count, 1, 1)
            same_fx_sensitivity = SensitivityConstaint(num_views, first, second)
            constraintMatrix.append(same_fx_sensitivity.generate_constaint_entry())

    constraintMatrix = np.array(constraintMatrix)
    jacobian = np.array(jacobian)

    Cprime = constraintMatrix @ jacobian

    # now I should be able to generate a dOmega
    # maybe distort by a small domega proportional to biggest value in DAQ?
    degenerate = False
    for _ in range(3):
        dOmega = np.random.uniform(-1e-4, 1e-4, 10)

        disturbed = Cprime @ dOmega  # one of these should be very different
        # from zero now if it is a unique solution
        maxDisturbed = np.max(disturbed)
        print(f"\tMax disturbed val: {maxDisturbed}")
        if abs(maxDisturbed) > 0.1:
            degenerate = True

    #for K in Ks:
    # if abs(K[0,1]) > 5:
    #print (f"\tK:\n{K}")

    print(f"\tIs this close to a degenerate configuration?: {degenerate}")

    return degenerate, maxDisturbed


def self_calibrate(projectionMatrices, imWidth, imHeight, groundTruth=flyingThingsIntrinsicMatrix):

    a = AutoCalibrationLinear()
    for i, pRaw in enumerate(projectionMatrices):
        # with 3 views I will have 2 sets of 4 useful equations...
        # therefore I do need around 3 views projectively reconstructed so I can get 8 useful equations...
        # I have  sets of a couple o equations...
        a.addProjection(pRaw, imWidth, imHeight)

    # get the rectifying transformation and the Dual Absolute Quadric (4x4 matrix, in metric case should be diag(I, 0))
    hComputed, DAQ = a.metricTransformation()

    Ks = []
    Rs = []
    Ts = []
    sensitivityPs = []
    Ps = []

    for i, pRaw in enumerate(projectionMatrices):
        Pmetric = pRaw @ hComputed  # undistort the cams
        # normedP = a.normalizeProjection(projectionMatrices[i], imWidth, imHeight)
        # denorm = a.create_normalizing_transformation(imWidth, imHeight)
        # KKT = normedP @ DAQ @ normedP.T
        #
        # K = np.linalg.cholesky(KKT).T
        # K /= K[-1, -1]
        # K = denorm @ K
        # print(f"K from Quadric:\n{K}")
        Kcomputed, Rotation, t = factor(Pmetric)

        Ks.append(Kcomputed)
        Rs.append(Rotation)
        Ts.append(t)
        Ps.append(Pmetric)
        # have to set the calibration to Identity so I can do sensitivity analysis...
        normalizedProjection = np.linalg.inv(Kcomputed) @ Pmetric
        #Kn, Rn, tn = factor(normalizedProjection)
        sensitivityPs.append(normalizedProjection)

    # DAQ brought to 1,1,1,0 like its supposed to be.
    transformedDAQ = np.linalg.inv(hComputed) @ DAQ @ np.linalg.inv(hComputed.T)

    isDegenerate, maxDisturbed = prepare_sensitivity(projectionMatrices, sensitivityPs, DAQ, Ks)

    return Ks, isDegenerate, maxDisturbed
