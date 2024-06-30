#! usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import scipy
import math

class Func_create:

    def rect_pulse(self, x, width):
        return 0.5 if -width / 2 <= x < width / 2 else 0

    def trapz_pulse(self, x, width):
        return 1 if -width / 2 <= x < width / 2 else 0

    def triang_pulse(self, x, width):
        if -width / 2 <= x < width / 2:
            return (x + width / 2) / width

    def sin_pulse(self, x, width):
        if -width / 2 <= x < width / 2:
            return 0.5 * (1 - math.cos(math.pi * (x + width / 2) / width))

    def cos_pulse(self, x, width):
        if -width / 2 <= x < width / 2:
            return 0.5 * (1 + math.cos(math.pi * (x + width / 2) / width))

    def Sa(self, x, tao):
        return np.sin(x*tao/2)/(x*tao/2)

