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


def int2hex(rgb_color: (int, int, int)) -> "":
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb_color)


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


def create_ramp_by_color(resolution: int,
                         color: (int, int, int),
                         l_range:(float, float)=(0.1, 0.8)):
    h, _, s = colorsys.rgb_to_hls(*int2float(color))
    l_ramp = np.linspace(l_range[1], l_range[0], resolution)
    return [float2int(colorsys.hls_to_rgb(h, l, s)) for l in l_ramp]


def map_color(arr: [], base_color=None, color_res=256) -> "":
    min_a, max_a = (min(arr), max(arr))

    def get_num_col_index(num):
        return int(round((num - min_a) / (max_a - min_a) * (color_res - 1)))

    arr_n = [get_num_col_index(c) for c in arr]
    if base_color:
        color_ramp = [int2hex(c) for c in create_ramp_by_color(color_res, base_color)]
    else:
        color_ramp = palettes.viridis(color_res)

    arr_colors = [color_ramp[i] for i in arr_n]
    return arr_colors
