import json
import math
import numpy as np

SPEED_OF_LIGHT = 299792458.0

DWT_FREQ_OFFSET_MULTIPLIER = (998.4e6/2.0/1024.0/131072.0)
DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 = (-1.0e6/6489.6e6)

S_PER_DT_TS = 1.0E-15 * 15650.0
METER_PER_DWT_TS = ((SPEED_OF_LIGHT * 1.0E-15) * 15650.0)

DBM_TWR_DIST_ADJUST = {
    -61: -11,
    -63: -10.5,
    -65: -10.0,
    -67: -9.3,
    -69: -8.2,
    -71: -6.9,
    -73: -5.1,
    -75: -2.7,
    -77: 0.0,
    -79: 2.1,
    -81: 3.5,
    -83: 4.2,
    -85: 4.9,
    -87: 6.2,
    -89: 7.1,
    -91: 7.6,
    -93: 8.1
}

def get_dist(pos_a, pos_b):
    pos_a = np.array(pos_a)
    pos_b = np.array(pos_b)
    return np.linalg.norm(pos_a - pos_b)

def pair_index(a, b):
    if a > b:
        return int((a * (a - 1)) / 2 + b)
    else:
        return pair_index(b, a)


def convert_ts_to_sec(ts):
    return ts*S_PER_DT_TS

def convert_sec_to_ts(sec):
    ts = sec / S_PER_DT_TS
    assert convert_ts_to_sec(ts)-sec < 0.00001
    return ts

def convert_ts_to_m(ts):
    return convert_ts_to_sec(ts)*SPEED_OF_LIGHT

def convert_m_to_ts(m):
    return convert_sec_to_ts(m/SPEED_OF_LIGHT)


def ci_to_rd(ci):
    return 1.0 - ci*(DWT_FREQ_OFFSET_MULTIPLIER * DWT_HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6)