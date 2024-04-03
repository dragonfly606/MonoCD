import unittest
import numpy as np

from wavedata.tools.core.integral_image_2d import IntegralImage2D


class IntegralImage2DTest(unittest.TestCase):

    def test_integral_image_2d(self):

        test_mat = np.ones((3, 3)).astype(np.float32)

        # Generate integral image
        integral_image = IntegralImage2D(test_mat)
        boxes = np.array([[0, 0, 1, 1],
                          [0, 0, 2, 2],
                          [0, 0, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(boxes)

        # First box case = should be 1*1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)
        # Second box case = should be 2*2*2 = 8
        self.assertTrue(occupancy_count[1] == 4)
        # Third box case = should be 3*3*3 = 27
        self.assertTrue(occupancy_count[2] == 9)

        boxes = np.array([[1, 1, 2, 2],
                          [1, 1, 3, 3]]).T.astype(np.uint32)

        occupancy_count = integral_image.query(boxes)

        # First box case = should be 1*1 = 1
        self.assertTrue(occupancy_count[0] == 1)

        # Second box case = should be 2*2 = 4
        self.assertTrue(occupancy_count[1] == 4)

        boxes = np.array([[0, 0, 3, 1]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(boxes)

        # Flat Surface case = should be 1*3 = 3
        self.assertTrue(occupancy_count[0] == 3)

        # Test outside the boundary
        boxes = np.array([[0, 0, 2312, 162]]).T.astype(np.uint32)
        occupancy_count = integral_image.query(boxes)
        self.assertTrue(occupancy_count[0] == 9)


if __name__ == '__main__':
    unittest.main()
