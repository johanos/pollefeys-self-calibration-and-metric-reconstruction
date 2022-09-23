import numpy as np
import numpy.linalg as LA


def vectorFromAbsoluteQuadric(Q):
    q = np.array([Q[0, 0], Q[0, 1], Q[0, 2], Q[0, 3], Q[1, 1], Q[1, 2], Q[1, 3], Q[2, 2], Q[2, 3], Q[3, 3]])
    return q


def absoluteQuadricMatFromVec(q):
    Q = np.array([[q[0], q[1], q[2], q[3]], [q[1], q[4], q[5], q[6]], [q[2], q[5], q[7], q[8]],
                  [q[3], q[6], q[8], q[9]]],
                 dtype=np.float32)

    return Q


class AutoCalibrationLinear(object):

    def __init__(self):
        self.projections = []  # the NORMALIZED projection matrices
        self.widths = []  # widths of images.. in case they differ.
        self.heights = []  # heights of images.. in case they differ.
        self.constraints = []  # linear constraints on q

    def addProjection(self, P, width, height):
        """
        :param P: Projection Matrix
        :param width: width of the image plane
        :param height: height of the image plane
        Width and height are to normalize the projection matrix for numerical stability. need not be exact.
        """

        pNormalized = self.normalizeProjection(P, width, height)
        self.addProjectionConstraints(pNormalized)

        # store input
        self.projections.append(pNormalized)
        self.widths.append(width)
        self.heights.append(height)

        return len(self.projections) - 1

    def metricTransformation(self):
        """
        Compute the metric updating transform
        :return: Homography H that transforms the space into metric space.
        if {P,X} is a projective reconstruction hen {PH, H^{-1} X} is a metric reconstruction.
        HZ section 19.1 page 459 not pollefeys.
        """
        # Compute the dual absolute quadric, Q

        A = np.zeros((len(self.constraints), 10))

        for i in range(A.shape[0]):
            A[i, :] = self.constraints[i]

        # solve the linear system Ax = 0 such that |x| = 1

        U, s, VT = np.linalg.svd(A, full_matrices=True)
        q = VT[-1, :]

        QStar = absoluteQuadricMatFromVec(q)

        # Force Rank 3...
        U, s_hat, VT = np.linalg.svd(QStar, full_matrices=True)
        s = np.zeros((4, 4))
        s[0][0] = s_hat[0]
        s[1][1] = s_hat[1]
        s[2][2] = s_hat[2]

        Q = np.dot(U, np.dot(s, VT))

        eigenValues, eigenVectors = LA.eigh(Q)

        # eigen values should be positive this makes it Positive definite...
        for i in range(len(eigenValues)):
            if eigenValues[i] < 0:
                eigenValues[i] = -eigenValues[i]

        # Q as Positive Definite
        QNow = eigenVectors @ np.diag(eigenValues) @ eigenVectors.T
        # and sorted so last one is 0...
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]

        # Compute the transformation from the eigen decomposition. Last paragraph of page 3 in
        # Autocalibration and the absolute quadric by B. Triggs.
        eigenValues[3] = 1
        eigenValues = np.sqrt(eigenValues)
        H = eigenVectors @ np.diag(eigenValues)

        return H, QNow

    def addProjectionConstraints(self, P):
        """
        Add constraints on the absolute quadric based assumptions on the parameters of one camera.
        :param P: the projection used to project the absolute quadric
        :return:
        """
        nu = 1

        # # non-extreme focal length.
        # self.constraints.append( (self.wc(P, 0, 0) - self.wc(P, 2, 2)) / (9 * nu) )
        # self.constraints.append( (self.wc(P, 1, 1) - self.wc(P, 2, 2)) / (9 * nu) )

        # # aspect ratio is near 1
        # TODO: figure out how to transform
        #self.constraints.append( (self.wc(P, 0, 0) - self.wc(P, 1, 1)) / ( 0.2 * nu) )
        self.constraints.append((self.wc(P, 0, 0) - self.wc(P, 1, 1)))

        # no skew and principal points near 0,0
        # self.constraints.append( self.wc(P, 0, 1) / ( 0.01 * nu))
        # self.constraints.append( self.wc(P, 0, 2) / ( 0.1 * nu))
        # self.constraints.append( self.wc(P, 1, 2) / ( 0.1 * nu))

        self.constraints.append(self.wc(P, 0, 1))
        self.constraints.append(self.wc(P, 0, 2))
        self.constraints.append(self.wc(P, 1, 2))

    def wc(self, P, i, j):
        """
        Computes the constraint associated with elements of the DIAC
        :param P: Projection matrix used to project the absolute quadric
        :param i: Row of the DIAC
        :param j: Column of the DIAC

        :return: the coefficients of the element i,j of the dual image of the absolute conic when written as a
        linear combination of the elements of the absolute quadric.
        There are 10 coefficients because quadric is 10 numbers
        """

        constraint = np.zeros(10, dtype=np.float64)

        for k in range(10):
            q = np.zeros(10, dtype=np.float64)
            q[k] = 1
            Q = absoluteQuadricMatFromVec(q)
            w = P @ Q @ P.T
            constraint[k] = w[i, j]

        return constraint

    def create_normalizing_transformation(self, width, height):

        T = np.array([[width + height, 0, (width - 1) / 2], [0, width + height, (height - 1) / 2], [0, 0, 1]],
                     dtype=np.float32)

        return T

    def normalizeProjection(self, P, width, height):
        """
        normalized projection matrix based on some assumptions
        :param P: Projection matrix
        :param width: width of the image
        :param height: height of the image
        :return: a normalized projection
        """

        T = self.create_normalizing_transformation(width, height)

        return np.matmul(LA.inv(T), P)

    def denormalizeProjection(self, P, width, height):
        """
        denormalize projection matrix based on same assumptions
        :param P:
        :param width:
        :param height:
        :return:
        """
        T = self.create_normalizing_transformation(width, height)
        return np.matmul(T, P)


def KfromAbsoluteConic(DIAC):
    """
    :param W: image of absolute conic....
    :return: Intrinsic camera parameters for a metric reconstruction.
    """

    IAC = LA.inv(DIAC)
    flippedIAC = np.zeros((3, 3), dtype=np.float32)
    for i in range(flippedIAC.shape[0]):
        for j in range(flippedIAC.shape[1]):
            flippedIAC[i, j] = IAC[2 - i, 2 - j]

    L = LA.cholesky(flippedIAC)
    K = np.zeros((3, 3), dtype=np.float32)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i, j] = L[2 - i, 2 - j]

    # resolve sign ambiguities assuming positive diagonal...

    for j in range(K.shape[0]):
        if K[j, j] < 0:
            for i in range(K.shape[1]):
                K[i, j] = -K[i, j]

    return K
