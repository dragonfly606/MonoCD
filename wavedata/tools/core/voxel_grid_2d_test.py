import os
import unittest
import numpy as np

from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

# ROOT_DIR at wavedata
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class VoxelGrid2DTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_points = np.array([[-39.99, 4.99, 0],
                                    [39.99, 4.99, 0],
                                    [-39.99, -4.99, 0],
                                    [39.99, -4.99, 0],
                                    [-39.99, 4.99, 69.99],
                                    [39.99, 4.99, 69.99],
                                    [-39.99, -4.99, 69.99],
                                    [39.99, -4.99, 69.99],
                                    [-39.99, 4.99, 69.99],
                                    [39.99, 4.99, 69.99],
                                    [-39.99, -4.99, 69.99],
                                    [39.99, -4.99, 69.99]])

        # Expected leaf layout for voxelization at size 0.1
        # y-axis for filled_indices should be 0
        filled_indices = \
            np.floor((cls.test_points * 10) + [400, 0, 0]).astype(np.int32)
        filled_indices[:, 1] = 0
        cls.expected_leaf_layout = VoxelGrid2D.VOXEL_EMPTY * \
            np.ones((800, 1, 700))
        for idx in filled_indices:
            cls.expected_leaf_layout[tuple(idx)] = VoxelGrid2D.VOXEL_FILLED

    def test_voxelization_2d(self):

        voxel_grid = VoxelGrid2D()

        # Test with actual data
        voxel_grid.voxelize_2d(self.test_points, 0.1)

        # Test Size variable
        self.assertAlmostEqual(voxel_grid.voxel_size, 0.1)

        # Test Minimum Coordinates
        self.assertTrue((voxel_grid.min_voxel_coord == [-400, 0, 0]).all())

        # Test Maximum Coordinates
        self.assertTrue((voxel_grid.max_voxel_coord == [399, 0, 699]).all())

        # Test Divisions
        self.assertTrue((voxel_grid.num_divisions == [800, 1, 700]).all())

        # Test every entry of out put leafs
        self.assertTrue((voxel_grid.leaf_layout_2d ==
                         self.expected_leaf_layout).all())

    def test_voxel_grid_2d_extents(self):

        voxel_grid = VoxelGrid2D()

        # Generate random points between xyz [[-40, 40], [-4, 4], [-30, 30]]
        points = (np.random.rand(70000, 3) * [80, 8, 60]) - [40, 4, 0]

        # Test bad extents
        bad_extents = np.array([[-30, 30], [-3, 3], [10, 60]])
        self.assertRaises(ValueError, voxel_grid.voxelize_2d, points, 0.1,
                          bad_extents)

        extents = np.array([[-50, 50], [-5, 5], [0, 70]])
        voxel_grid.voxelize_2d(points, 0.1, extents)

        # Check number of divisions and leaf layout shape are correct and are
        # the same. y-axis shapes should be 1
        self.assertTrue((voxel_grid.num_divisions == [1000, 1, 700]).all())
        self.assertTrue(voxel_grid.leaf_layout_2d.shape == (1000, 1, 700))

    def test_voxel_2d_coordinate_conversion(self):
        voxel_grid = VoxelGrid2D()

        # Generate random points between xyz [[-40, 40], [-4, 4], [-30, 30]]
        points = (np.random.rand(70000, 3) * [80, 8, 60]) - [40, 4, 0]
        extents = np.array([[-50, 50], [-5, 5], [0, 70]])
        voxel_grid.voxelize_2d(points, 0.1, extents)

        # Left Top Corner, z = 0
        coordinates = np.array([[0, 0]])
        # Map spans from [-500, 500], [0, 700]
        expected = np.array([500, 0])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        coordinates = np.array([[0, 0]]) + 0.1
        # Increment of 1 grid size
        expected = np.array([501, 1])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # Start of Grid
        coordinates = np.array([[-50, 0]])
        expected = np.array([0, 0])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # End of Grid
        coordinates = np.array([[50, 70]])
        expected = np.array([1000, 700])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # Outside the grid
        coordinates = coordinates + 10
        expected = np.array([1000, 700])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())


if __name__ == '__main__':
    unittest.main()
