"""
Includes color-related functions
From toy_data:
    Module for toy-data generation for ML experiments
"""

import colorsys
import numpy.random as rnd

color_loop = [
    (160, 124, 58),
    (123, 70, 195),
    (89, 152, 62),
    (196, 76, 165),
    (73, 100, 60),
    (207, 74, 104),
    (72, 137, 159),
    (206, 85, 47),
    (124, 120, 191),
    (125, 63, 62),
    (73, 42, 94)
]


def get_N_by_hue(N, s=0.7, v=0.7):
    phase = rnd.random()
    HSV_tuples = list()
    for x in range(N):
        hue = (((x + rnd.random()/2) / N) + phase) % 1
        HSV_tuples.append((hue, s, v))

    colors = []
    for hsv in HSV_tuples:
        c = tuple(int(round(255 * rgb)) for rgb in colorsys.hsv_to_rgb(*hsv))
        colors.append(c)
    return colors


def random_color(v=0.7) -> (int, int, int):
    hsv = (rnd.random(), rnd.random(), v)
    c = tuple(int(round(255 * rgb)) for rgb in colorsys.hsv_to_rgb(*hsv))
    return c

