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
    

class BoundingBox3D:

    def __init__(self) -> None:
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None
        self.x_center = None
        self.y_center = None
        self.z_center = None
        self.x_size   = None
        self.y_size   = None
        self.z_size   = None

    @classmethod
    def from_center_and_size(cls, x_center : float, y_center : float, z_center : float, x_size : float, y_size : float, z_size : float ) -> "BoundingBox3D":
        box = BoundingBox3D()
        box.x_center = x_center
        box.y_center = y_center
        box.z_center = z_center
        box.x_size = x_size
        box.y_size = y_size
        box.z_size = z_size
        box.x_min = x_center - (x_size/2)
        box.x_max = x_center + (x_size/2)
        box.y_min = y_center - (y_size/2)
        box.y_max = y_center + (y_size/2)
        box.z_min = z_center - (z_size/2)
        box.z_max = z_center + (z_size/2)
        return box

    @classmethod
    def from_corners(cls, x_min : float, y_min : float, z_min : float, x_max : float, y_max : float, z_max : float) -> "BoundingBox3D":
        box = BoundingBox3D()
        box.x_center = (x_max + x_min) * 0.5
        box.y_center = (y_max + y_min) * 0.5
        box.z_center = (z_max + z_min) * 0.5
        box.x_size = x_max - x_min
        box.y_size = y_max - y_min
        box.z_size = z_max - z_min
        box.x_min = x_min
        box.x_max = x_max
        box.y_min = y_min
        box.y_max = y_max
        box.z_min = z_min
        box.z_max = z_max
        return box

    def get_corners(self) -> Tuple[float, float, float, float, float, float]:
        """Returns the [x_min,y_min,z_min,x_max,y_max,z_max] coordinates of the box."""
        return [self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max]

    def get_as_center_and_size(self) -> Tuple[float, float, float, float, float, float]:
        """Returns the [x_center, y_center, z_center, x_size, y_size, z_size] of the box."""
        return [self.x_center, self.y_center, self.z_center, self.x_size, self.y_size, self.z_size]

    def to_bounding_box_2D(self) -> BoundingBox2D:
        box = BoundingBox2D.from_corners(self.x_min, self.y_min, self.x_max, self.y_max)
        return box