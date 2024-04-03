"""This module converts given point clouds to 3D voxel grid."""
import numpy as np


class VoxelGrid(object):
    """Voxel Grid Class Object
    - Voxels are 3D grids that represent occupancy info.
    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be -1 if free/occluded
    and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    div_multiplier: multiplier to each iterator (i.e. i,j,k) to obtain grid index in 1D
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied
    """

    # Class Constants
    VOXEL_EMPTY = -1
    VOXEL_FILLED = 0

    def __init__(self):

        self.voxel_size = 0.0
        self.min_voxel_coord = []
        self.max_voxel_coord = []
        self.num_divisions = [0, 0, 0]

        self.points = []
        self.voxel_indices = []
        self.leaf_layout = []

    def voxelize(self, pts, voxel_size, extents=None, create_leaf_layout=True):
        """
        The input for the voxelization is expected to be a PointCloud
        with N points in 3 dimension (x,y,z). Voxel size is the quantization size
        for the voxel grid.

        :param pts: Point cloud as N x [x, y, z]
        :param voxel_size: Quantization size for the grid
        :param extents: Optional, specifies the full extents of the point cloud.
                        Used for creating same sized voxel grids.
        :param create_leaf_layout: Set this to False to create an empty leaf_layout,
                                   which will save computation time.
        """
        # Check if points are 3D, otherwise early exit
        if pts.shape[1] != 3:
            raise ValueError("Points have the wrong shape: {}".format(pts.shape))

        self.voxel_size = voxel_size

        # Discretize voxel coordinates to given quantization size
        discrete_pts = np.floor(pts/voxel_size).astype(np.int32)

        # Use Lex Sort, sort by x, then y, then z
        x_col = discrete_pts[:, 0]
        y_col = discrete_pts[:, 1]
        z_col = discrete_pts[:, 2]
        sorted_order = np.lexsort((z_col, y_col, x_col))

        # Save original points in sorted order
        self.points = pts[sorted_order]
        discrete_pts = discrete_pts[sorted_order]

        # Format the array to c-contiguous array for unique function
        contiguous_array = np.ascontiguousarray(discrete_pts).view(
                            np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

        # The new coordinates are the discretized array with its unique indexes
        _, unique_indices = np.unique(contiguous_array, return_index=True)

        # Sort unique indices to preserve order
        unique_indices.sort()

        voxel_coords = discrete_pts[unique_indices]

        # Find the minimum and maximum voxel coordinates
        if extents is not None:
            # Check provided extents
            extents_transpose = np.array(extents).transpose()
            if extents_transpose.shape != (2, 3):
                raise ValueError("Extents are the wrong shape {}".format(extents.shape))

            # Set voxel grid extents
            self.min_voxel_coord = np.floor(extents_transpose[0] / voxel_size)
            self.max_voxel_coord = np.ceil(extents_transpose[1] / voxel_size) - 1

            # Check that points are bounded by new extents
            if not (self.min_voxel_coord <= np.amin(voxel_coords, axis=0)).all():
                raise ValueError("Extents are smaller than min_voxel_coord")
            if not (self.max_voxel_coord >= np.amax(voxel_coords, axis=0)).all():
                raise ValueError("Extents are smaller than max_voxel_coord")
        else:
            # Automatically calculate extents
            self.min_voxel_coord = np.amin(voxel_coords, axis=0)
            self.max_voxel_coord = np.amax(voxel_coords, axis=0)

        # Get the voxel grid dimensions
        self.num_divisions = ((self.max_voxel_coord - self.min_voxel_coord) + 1).astype(np.int32)

        # Bring the min voxel to the origin
        self.voxel_indices = (voxel_coords - self.min_voxel_coord).astype(int)

        if create_leaf_layout:
            # Create Voxel Object with -1 as empty/occluded
            self.leaf_layout = self.VOXEL_EMPTY * \
                               np.ones(self.num_divisions.astype(int))

            # Fill out the leaf layout
            self.leaf_layout[self.voxel_indices[:, 0],
                             self.voxel_indices[:, 1],
                             self.voxel_indices[:, 2]] = \
                self.VOXEL_FILLED

    def map_to_index(self, map_index):
        """ convert map coordinate values to 1-based discrectized grid index coordinate
            Note: Any values outside the extent of the grid will be forced to be the maximum grid
            coordinate.

        :param map_index: N x 3 points

        :return: N x length(dim) (grid coordinate)
                [] if min_voxel_coord or voxel_size or grid_index or dim is not set.
        """
        if self.voxel_size == 0 \
                or len(self.min_voxel_coord) == 0 \
                or len(map_index) == 0:
            return []

        return np.maximum(0, np.minimum(self.num_divisions[:], np.floor(map_index/self.voxel_size)
                                        - self.min_voxel_coord[:]))
