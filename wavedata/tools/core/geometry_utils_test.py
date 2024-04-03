import numpy as np
import unittest

from wavedata.tools.core import geometry_utils


class GeometryUtilsTest(unittest.TestCase):

    def test_dist_to_plane(self):

        xy_plane = [0, 0, 1, 0]
        xz_plane = [0, 1, 0, 0]
        yz_plane = [1, 0, 0, 0]
        diagonal_plane = [1, 1, 1, 0]

        point = [[1, 1, 1]]

        dist_from_xy = geometry_utils.dist_to_plane(xy_plane, point)
        dist_from_xz = geometry_utils.dist_to_plane(xz_plane, point)
        dist_from_yz = geometry_utils.dist_to_plane(yz_plane, point)
        dist_from_diag = geometry_utils.dist_to_plane(diagonal_plane, point)

        self.assertAlmostEqual(dist_from_xy[0], 1.0)
        self.assertAlmostEqual(dist_from_xz[0], 1.0)
        self.assertAlmostEqual(dist_from_yz[0], 1.0)
        self.assertAlmostEqual(dist_from_diag[0], np.sqrt(3))

        # Check that a signed distance is returned
        xy_plane_inv = [0, 0, -1, 0]
        diagonal_plane_inv = [-1, -1, -1, 0]

        dist_from_xy_inv = geometry_utils.dist_to_plane(xy_plane_inv, point)
        dist_from_diag_inv = geometry_utils.dist_to_plane(diagonal_plane_inv,
                                                          point)

        self.assertAlmostEqual(dist_from_xy_inv[0], -1.0)
        self.assertAlmostEqual(dist_from_diag_inv[0], -np.sqrt(3))


if __name__ == '__main__':
    unittest.main()
