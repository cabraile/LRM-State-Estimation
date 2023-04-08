from typing import Tuple

class BoundingBox2D:

    def __init__(self) -> None:
        self.x_left   = None
        self.x_right  = None
        self.y_top    = None
        self.y_bottom = None
        self.x_center = None
        self.y_center = None
        self.x_size   = None
        self.y_size   = None

    @classmethod
    def from_center_and_size(cls, x_center : float, y_center : float, x_size : float, y_size : float ) -> "BoundingBox2D":
        box = BoundingBox2D()
        box.x_center = x_center
        box.y_center = y_center
        box.x_size = x_size
        box.y_size = y_size
        box.x_left = x_center - (x_size/2)
        box.x_right = x_center + (x_size/2)
        box.y_bottom = y_center - (y_size/2)
        box.y_top = y_center + (y_size/2)
        return box

    @classmethod
    def from_corners(cls, x_left : float, y_bottom : float, x_right : float, y_top : float) -> "BoundingBox2D":
        box = BoundingBox2D()
        box.x_center = (x_right + x_left) * 0.5
        box.y_center = (y_top + y_bottom) * 0.5
        box.x_size = x_right - x_left
        box.y_size = y_top - y_bottom
        box.x_left   = x_left
        box.x_right  = x_right
        box.y_bottom = y_bottom
        box.y_top    = y_top
        return box

    def get_corners(self) -> Tuple[float,float, float, float]:
        """Returns the [left,bottom,right,top] coordinates of the box."""
        return [self.x_left, self.y_bottom, self.x_right, self.y_top]

    def get_as_center_and_size(self) -> Tuple[float, float, float, float]:
        """Returns the [x_center, y_center, x_size, y_size] of the box."""
        return [self.x_center, self.y_center, self.x_size, self.y_size]