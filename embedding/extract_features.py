"""
Color features extraction
Made by Lee Se Hyeon 22th Jan 2024
"""

import itertools
import numpy as np
import colorsys

def rgb_to_lab(r, g, b):
    # Convert RGB to XYZ
    r = 100.0 * (r / 255.0)
    g = 100.0 * (g / 255.0)
    b = 100.0 * (b / 255.0)

    r = r / 100.0
    g = g / 100.0
    b = b / 100.0

    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92

    r = r * 100.0
    g = g * 100.0
    b = b * 100.0

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Convert XYZ to LAB
    x /= 95.047
    y /= 100.000
    z /= 108.883

    if x > 0.008856:
        x = x ** (1.0 / 3.0)
    else:
        x = (903.3 * x + 16.0) / 116.0

    if y > 0.008856:
        y = y ** (1.0 / 3.0)
    else:
        y = (903.3 * y + 16.0) / 116.0

    if z > 0.008856:
        z = z ** (1.0 / 3.0)
    else:
        z = (903.3 * z + 16.0) / 116.0

    l = max(0.0, min(100.0, (116.0 * y) - 16.0))
    a = max(-128.0, min(127.0, (x - y) * 500.0))
    b = max(-128.0, min(127.0, (y - z) * 200.0))

    return [l, a, b]


def calculate_statistics(channel):
    mean_value = np.mean(channel)
    stddev_value = np.std(channel)
    median_value = np.median(channel)
    max_value = np.max(channel)
    min_value = np.min(channel)
    maxmin_diff = np.max(channel) - np.min(channel)
    return [round(mean_value, 3), round(stddev_value, 3), round(median_value, 3), round(max_value, 3), round(min_value, 3), round(maxmin_diff, 3)]


def calculate_hue_probability(palette):
    probabilities = []
    for pal_hue in palette:
        probabilities.append([np.abs(pal_hue - hue) for hue in palette])
    return np.round(probabilities, 3).tolist()


def calculate_hue_entropy(palette):
    normalized_hues = palette / np.sum(palette)

    non_zero_normalized_hues = normalized_hues[normalized_hues > 0]

    # Calculate entropy using the formula: -sum(p_i * log2(p_i))
    entropy = -np.sum(non_zero_normalized_hues * np.log2(non_zero_normalized_hues))

    return entropy


def calculate_CH_values(L1, C1, H1, L2, C2, H2):
    # Calculate HC
    delta_C = np.sqrt((np.square(H1 - H2) + np.square(C1 - C2 / 1.46)) / 2)
    HC = 0.04 + 0.53 * np.tanh(0.8 - 0.045 * delta_C)

    # Calculate HL
    Lsum = L1 + L2
    HLsum = 0.28 + 0.54 * np.tanh(-3.88 + 0.029 * Lsum)
    delta_L = np.abs(L1 - L2)
    Hdelta_L = 0.14 + 0.15 * np.tanh(-2 + 0.2 * delta_L)
    HL = HLsum + Hdelta_L

    # Calculate HH
    EC = 0.5 + 0.5 * np.tanh(-2 + 0.5 * C1)
    HS = -0.08 - 0.14 * np.sin(np.radians(H1) + np.radians(50)) - 0.07 * np.sin(np.radians(2 * H1) + np.radians(90))
    HSY1 = EC * (HS + L1)
    HS = -0.08 - 0.14 * np.sin(np.radians(H2) + np.radians(50)) - 0.07 * np.sin(np.radians(2 * H2) + np.radians(90))
    HSY2 = EC * (HS + L2)
    HH = HSY1 + HSY2

    # Calculate CH
    CH = HC + HL + HH

    return round(CH, 3)


def calculate_CH(palette):
    ch_values = []
    for color1, color2 in itertools.combinations(palette, 2):
        l1, a1, b1 = color1
        l2, a2, b2 = color2
        result = calculate_CH_values(l1, np.sqrt(a1**2 + b1**2), np.degrees(np.arctan2(b1, a1)),
                           l2, np.sqrt(a2**2 + b2**2), np.degrees(np.arctan2(b2, a2)))
        ch_values.append(result)
    return ch_values


def calculate_gradient(hue_values):
    hue_gradient = np.diff(hue_values)

    return np.round(hue_gradient, 3).tolist()

def get_features(colors_4dim):
    palette = colors_4dim

    palette_RGB = []
    palette_HSV = []
    palette_LAB = []

    for color in palette:
        color = [round(c / 255.0, 3) for c in color]
        palette_RGB.append(color)
        palette_HSV.append([round(i, 3) for i in list(colorsys.rgb_to_hsv(*color))])
        palette_LAB.append([round(i, 3) for i in rgb_to_lab(*color)])

    statistics_RGB_R = [calculate_statistics(np.array(palette_RGB)[:,0])]
    statistics_RGB_G = [calculate_statistics(np.array(palette_RGB)[:,1])]
    statistics_RGB_B = [calculate_statistics(np.array(palette_RGB)[:,2])]
    statistics_HSV_H = [calculate_statistics(np.array(palette_HSV)[:,0])]
    statistics_HSV_S = [calculate_statistics(np.array(palette_HSV)[:,1])]
    statistics_HSV_V = [calculate_statistics(np.array(palette_HSV)[:,2])]
    statistics_LAB_L = [calculate_statistics(np.array(palette_LAB)[:,0])]
    statistics_LAB_A = [calculate_statistics(np.array(palette_LAB)[:,1])]
    statistics_LAB_B = [calculate_statistics(np.array(palette_LAB)[:,2])]

    hue_array = np.array(palette_HSV)[:,0]
    hue_probabilities = calculate_hue_probability(hue_array)
    hue_entropy = [[calculate_hue_entropy(hue_array)]]

    ch_values = [calculate_CH(np.array(palette_LAB))]
    hue_gradient = [calculate_gradient(hue_array)]
    light_gradient = [calculate_gradient(np.array(palette_LAB)[:,0])]

    out_features = sum((palette_RGB, palette_HSV, palette_LAB,
                        statistics_RGB_R, statistics_RGB_G, statistics_RGB_B,
                        statistics_HSV_H, statistics_HSV_S, statistics_HSV_V,
                        statistics_LAB_L, statistics_LAB_A, statistics_LAB_B,
                        hue_probabilities, hue_entropy,
                        ch_values, hue_gradient, light_gradient), [])

    return sum(out_features, [])

if __name__ == "__main__":
    
    colors_4dim = [[0, 73, 83], [154, 205, 50], [209, 226, 49], [238, 220, 130]]
    features = get_features(colors_4dim)
    print(features)