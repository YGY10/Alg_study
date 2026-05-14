import math
import numpy as np


class BEVGrid:
    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)

        self.width = int(math.ceil((self.y_max - self.y_min) / self.resolution))
        self.height = int(math.ceil((self.x_max - self.x_min) / self.resolution))

        self._points_vehicle = None

    def get_size(self):
        return self.width, self.height

    def generate_ground_points_vehicle(self):
        """
        Return ground points for each BEV pixel in ego vehicle frame.

        Shape:
            points: (H * W, 3)

        Vehicle frame:
            x forward
            y right
            z up

        BEV image:
            row=0 means x=x_max
            row=H-1 means x=x_min
            col=0 means y=y_min
            col=W-1 means y=y_max
        """
        if self._points_vehicle is not None:
            return self._points_vehicle

        rows = np.arange(self.height, dtype=np.float64)
        cols = np.arange(self.width, dtype=np.float64)

        # Pixel center coordinates
        x_values = self.x_max - (rows + 0.5) * self.resolution
        y_values = self.y_min + (cols + 0.5) * self.resolution

        yy, xx = np.meshgrid(y_values, x_values)

        zz = np.zeros_like(xx)

        points = np.stack(
            [
                xx.reshape(-1),
                yy.reshape(-1),
                zz.reshape(-1),
            ],
            axis=1,
        )

        self._points_vehicle = points
        return points

    def vehicle_xy_to_bev_pixel(self, x, y):
        """
        Convert ego vehicle xy coordinate to BEV pixel row/col.
        """
        col = (y - self.y_min) / self.resolution
        row = (self.x_max - x) / self.resolution
        return row, col

    def bev_pixel_to_vehicle_xy(self, row, col):
        """
        Convert BEV pixel row/col to ego vehicle xy coordinate.
        """
        x = self.x_max - (row + 0.5) * self.resolution
        y = self.y_min + (col + 0.5) * self.resolution
        return x, y