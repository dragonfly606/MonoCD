import os
import unittest
import numpy as np

from wavedata.tools.core.voxel_grid import VoxelGrid

# ROOT_DIR at wavedata
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class VoxelGridTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_points = np.array(
            [[-39.99, 4.99, 0],
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
        filled_indices = np.floor(
            (cls.test_points * 10) + [400, 50, 0]).astype(np.int32)
        cls.expected_leaf_layout = VoxelGrid.VOXEL_EMPTY * \
            np.ones((800, 100, 700))
        for idx in filled_indices:
            cls.expected_leaf_layout[tuple(idx)] = VoxelGrid.VOXEL_FILLED

    def test_voxelization(self):

        voxel_grid = VoxelGrid()

        # Test with actual data
        voxel_grid.voxelize(self.test_points, 0.1)

        # Test Size variable
        self.assertAlmostEquals(voxel_grid.voxel_size, 0.1)

        # Test Minimum Coordinates
        self.assertTrue((voxel_grid.min_voxel_coord == [-400, -50, 0]).all())

        # # Test Maximum Coordinates
        self.assertTrue((voxel_grid.max_voxel_coord == [399, 49, 699]).all())

        # Test Divisions
        self.assertTrue((voxel_grid.num_divisions == [800, 100, 700]).all())

        # Test every entry of out put leafs
        self.assertTrue(
            (voxel_grid.leaf_layout == self.expected_leaf_layout).all())

    def test_voxel_grid_extents(self):

        voxel_grid = VoxelGrid()

        # Generate random points between xyz [[-40, 40], [-4, 4], [-30, 30]]
        points = (np.random.rand(70000, 3) * [80, 8, 60]) - [40, 4, 0]

        # Test bad extents
        bad_extents = np.array([[-30, 30], [-3, 3], [10, 60]])
        self.assertRaises(
            ValueError,
            voxel_grid.voxelize,
            points,
            0.1,
            bad_extents)

        extents = np.array([[-50, 50], [-5, 5], [0, 70]])
        voxel_grid.voxelize(points, 0.1, extents)

        # Check number of divisions and leaf layout shape are correct and are
        # the same
        self.assertTrue((voxel_grid.num_divisions == [1000, 100, 700]).all())
        self.assertTrue(voxel_grid.leaf_layout.shape == (1000, 100, 700))

    def test_voxel_coordinate_conversion(self):
        voxel_grid = VoxelGrid()

        # Generate random points between xyz [[-40, 40], [-4, 4], [-30, 30]]
        points = (np.random.rand(70000, 3) * [80, 8, 60]) - [40, 4, 0]
        extents = np.array([[-50, 50], [-5, 5], [0, 70]])
        voxel_grid.voxelize(points, 0.1, extents)

        # Left Top Corner, z = 0
        coordinates = np.array([[0, 0, 0]])
        # Map spans from [-500, 500], [-50, 50], [0, 700]
        expected = np.array([500, 50, 0])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        coordinates = np.array([[0, 0, 0]]) + 0.1
        # Increment of 1 grid size
        expected = np.array([501, 51, 1])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # Start of Grid
        coordinates = np.array([[-50, -5, 0]])
        expected = np.array([0, 0, 0])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # End of Grid
        coordinates = np.array([[50, 5, 70]])
        expected = np.array([1000, 100, 700])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())

        # Outside the grid
        coordinates = coordinates + 10
        expected = np.array([1000, 100, 700])
        self.assertTrue(
            (voxel_grid.map_to_index(coordinates) == expected).all())


if __name__ == '__main__':
    unittest.main()
