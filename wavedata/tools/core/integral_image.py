""" This module provides an interface to calculate values inside a box given
    a set of coordinates and an image """
import os
import ctypes as ct
import numpy as np


class IntegralImage(object):
    """Class to handle cumulative summed table along with 3D integral image"""

    def __init__(self, img):
        # initialize with the 3d integral image
        self._integral_image = self._integral_image_3d(img)

        # keep map sizes
        self._x_size = self._integral_image.shape[0]
        self._y_size = self._integral_image.shape[1]
        self._z_size = self._integral_image.shape[2]

        # set ctype function handler
        current_file_dir = os.path.dirname(__file__)
        self._lib = ct.cdll.LoadLibrary(current_file_dir +
                                        '/lib/libintegral_images_3d.so')

    def _integral_image_3d(self, img):
        """Calculates a 3D integral image from an input image.

        :param img :    W x H x L array
                        Integral image of size W x H x L

        :return rt_image : IntegralImage object
                            Object containing integral image and its parameters.
                            Returns empty list on failure.
        """
        # Check if points are 3D otherwise early exit
        if img.ndim != 3:
            raise ValueError('Not a 3D image for integral image: input dim {}'
                             .format(img.ndim))

        integral_image = np.cumsum(np.cumsum(np.cumsum(img, 0), 1), 2)
        # pad integral image with 0s on one side of each dimension
        # so that when accessing coordinate n-1, we get a valid value of 0
        integral_image = np.pad(integral_image, ((1, 0), (1, 0), (1, 0)),
                                'constant', constant_values=0)

        # Convert to fortran style array for ctype function call for query
        integral_image = np.asfortranarray(integral_image, dtype=np.float32)

        return integral_image

    def query(self, cuboids):
        """Input is an array of 3D cuboids 6 coordinates. Each column
        represents a cuboid in the format [x1, y1, z1, x2, y2, z2].T. Thus,
        the dimensions should be 6 x N. The 2 sets of 3D coordinates represent
        the 2 corners of the bounding box. The first set of coordinates is the
        point closest to the origin of the image. The second set of coordinates
        is the point farthest from the origin. img is the integral image array.

        :param cuboids : 6 x N ndarray
            Contains the (x1, y1, z1) and (x2, y2, z2) coordinates
                            of the box to query.

        :return param : N x 1 ndarray
            List consists of values contained inside box specified by
            coordinates from cuboids. Empty on failure.
        """
        cuboids = np.asarray(cuboids)

        # check size
        if cuboids.shape[0] != 6:
            raise ValueError(
                'Incorrect number of dimensions for query: input dim {}'.format(cuboids.shape[0]))

        if cuboids.shape[1] < 1:
            raise ValueError(
                'The dimension N must be greater than 1: input dim {}'.format(cuboids.shape[1]))

        if cuboids.dtype != np.uint32:
            raise TypeError('Cuboids must be type of np.uint32')

        # Convert given array to a fortran contiguous array with dtype uint32
        # Add 1 for first 3 rows to account for zero-padding in first coordinate
        cuboids[:3, :] += 1

        cuboids = np.asfortranarray(cuboids)

        # Clip all the maximum coordinates to the voxelgrid size
        # Note: The integral image gets zero padded.
        max_extents = np.array(
            [self._x_size, self._y_size, self._z_size,
             self._x_size, self._y_size, self._z_size]) - 1

        cuboids = np.minimum(cuboids, max_extents.reshape(6, -1)) \
            .astype(np.uint32)

        int_img_fnc = self._lib.integralImage3D
        int_img_fnc.restypes = None
        int_img_fnc.argtypes = [
                                # list that stores outputs
                                np.ctypeslib.ndpointer(dtype=np.float32,
                                                       flags='C_CONTIGUOUS'),
                                # list of box coordinates
                                np.ctypeslib.ndpointer(dtype=np.uint32,
                                                       flags='F_CONTIGUOUS'),
                                ct.c_uint,  # number of boxes
                                # integral image
                                np.ctypeslib.ndpointer(dtype=np.float32,
                                                       flags='F_CONTIGUOUS'),
                                ct.c_uint,  # width of integral image
                                ct.c_uint,  # height of integral image
                                ct.c_uint,  # length of integral image
                               ]

        num_of_cuboids = cuboids.shape[1]

        # initialize output array
        output = np.empty((num_of_cuboids, 1), dtype=np.float32, order='C')

        int_img_fnc(output, cuboids, ct.c_uint(num_of_cuboids), self._integral_image,
                    ct.c_uint(self._x_size), ct.c_uint(self._y_size), ct.c_uint(self._z_size))

        return output
