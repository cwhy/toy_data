"""
Includes color-related functions
From toy_data:
    Module for toy-data generation for ML experiments
"""

import colorsys
import numpy as np
import numpy.random as rnd
import bokeh.palettes as palettes

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


def float2int(color_float: (float, float, float)) -> (int, int, int):
    return tuple(int(round(255 * rgb)) for rgb in color_float)


def int2float(color_int: (int, int, int)) -> (float, float, float):
    return tuple(rgb / 255 for rgb in color_int)


def get_N_by_hue(N, s=0.7, v=0.7):
    phase = rnd.random()
    HSV_tuples = list()
    for x in range(N):
        hue = (((x + rnd.random() / 2) / N) + phase) % 1
        HSV_tuples.append((hue, s, v))

    colors = [float2int(colorsys.hsv_to_rgb(*hsv)) for hsv in HSV_tuples]
    return colors


def random_color(v=0.7) -> (int, int, int):
    hsv = (rnd.random(), rnd.random(), v)
    return float2int(colorsys.hsv_to_rgb(*hsv))


def create_ramp_by_color(resolution: int, color: (int, int, int)):
    h, s, _ = colorsys.rgb_to_hsv(*int2float(color))
    v_ramp = np.linspace(0, 1, resolution)
    return [float2int(colorsys.hsv_to_rgb(h, s, v)) for v in v_ramp]


def map_color(arr, resolution, base_color=None):
    min_a, max_a = (min(arr), max(arr))

    def get_num_col_index(num):
        return int(round((num - min_a) / (max_a - min_a) * (resolution-1)))

    arr_n = [get_num_col_index(c) for c in arr]
    if base_color:
        color_ramp = create_ramp_by_color(resolution, base_color)
    else:
        color_ramp = palettes.viridis(resolution)

    arr_colors = [color_ramp[i] for i in arr_n]
    return arr_colors
