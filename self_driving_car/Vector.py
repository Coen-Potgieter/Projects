import math
import random

# --------------------------------defenitions------------------------------------ #


def dist(coord1, coord2):
    ph_step = pow((coord1[0] - coord2[0]), 2) + \
        pow((coord1[1] - coord2[1]), 2)

    return math.sqrt(ph_step)


def directions(size):

    return [Vector(magnitude=2, direction=random.uniform(0, 2*math.pi))
            for i in range(size)]

# --------------------------------classes------------------------------------ #


class Vector:
    def __init__(self, magnitude, direction):
        self.mag = magnitude
        self.direction = direction

    def _xcomp(self):
        return math.cos(self.direction)*self.mag

    def _ycomp(self):
        return math.sin(self.direction)*self.mag

    def __add__(self, other):

        new_x = self._xcomp() + other._xcomp()
        new_y = self._ycomp() + other._ycomp()
        new_mag = ((new_x**2) + (new_y**2))**(0.5)
        try:
            local_tri_angle = math.atan(abs(new_y)/abs(new_x))
        except ZeroDivisionError:
            local_tri_angle = 0

        if new_x > 0 and new_y > 0:    # first quad
            new_direction = local_tri_angle

        elif new_x < 0 and new_y > 0:   # second quad
            new_direction = math.pi - local_tri_angle

        elif new_x < 0 and new_y < 0:   # third quad
            new_direction = math.pi + local_tri_angle

        else:   # fourth quad
            new_direction = -local_tri_angle

        return Vector(new_mag, new_direction)
