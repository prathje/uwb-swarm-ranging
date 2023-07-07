

name = 'trento_a'

devs = [
    'dwm1001.1',
    'dwm1001.2',
    'dwm1001.3',
    'dwm1001.4',
    'dwm1001.5',
    'dwm1001.6',
    'dwm1001.7'
]

dev_ids = {
    'dwm1001.1': '0x5b9a',
    'dwm1001.2': '0x56d4',
    'dwm1001.3': '0x0345',
    'dwm1001.4': '0x9535',
    'dwm1001.5': '0x87e8',
    'dwm1001.6': '0xa7d8',
    'dwm1001.7': '0x24f1',
}

dev_positions = {
    'dwm1001.1': (76, 3.97, 0.0),
    'dwm1001.2': (72.74, 6.6, 0.0),
    'dwm1001.3': (75.97, 6.86, 0.0),
    'dwm1001.4': (78.98, 6.85, 0.0),
    'dwm1001.5': (78.89, 0.44, 0.0),
    'dwm1001.6': (75.89, 0.43, 0.0),
    'dwm1001.7': (72.91, 0.37, 0.0)
}

factory_delays = {
    'dwm1001.1': 16472,
    'dwm1001.2': 16459,
    'dwm1001.3': 16472,
    'dwm1001.4': 16472,
    'dwm1001.5': 16458,
    'dwm1001.6': 16460,
    'dwm1001.7': 16460
}

def parse_messages_from_lines(line_it, src_dev=None):
    import json

    dev_set = set(devs)


    if src_dev is not None:
        dev_set = {src_dev}

    for line in line_it:
        if line.strip() == "":
            continue
        try:

            #[2023-02-03 17:48:15,819] INFO:dwm1001.169: 169.dwm1001 < b'{"type": "drift_estimation", "round": 1, "durations": [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]]}'

            ts_str = line[1:24]

            dev = line[24:].split(':', 2)[1]

            if dev not in dev_set:
                continue

            json_str = line[24:].split(" < b\'", 2)[1][:-2]
            log_ts = ts_str

            try:
                msg = json.loads(json_str)
                msg['_log_ts'] = ts_str
                yield (log_ts, dev, msg)
            except json.decoder.JSONDecodeError:
                #print(json_str)
                pass
        except ValueError:
            pass


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

import numpy as np
from PIL import Image

def draw_layout(plt):



    lineargs = {
        "color": "black",
        #"alpha": 0.5,
        "lw": 1.5,
        "zorder": -1,
    }

    rectargs = {
        "facecolor": "white",
        "edgecolor": "black",
        #"alpha": 0.5,
        "lw": 1.5,
        "zorder": -1,
    }

    # npimage = np.flip(np.asarray(Image.open('img/trento_a.png')), axis=0)
    # scalingx = 0.01295
    # scalingy = 0.0134
    # tx = 72-0.561
    # ty = -1.5+1.024-0.535
    #plt.gca().imshow(npimage, origin="lower", extent=(tx, tx + npimage.shape[0]*scalingx, ty, ty + npimage.shape[1]*scalingy), zorder=-1)


    lines = [
        ((70.0, 2.45), (72.16, 2.45)),
        ((72.16, 2.45), (72.16, 8.5)),
        ((72.16, 6.37), (70.0, 6.37)),
        ((72.16, 7.41), (70.0, 7.41)),
        ((74.72, 7.41), (74.72, 8.5)),
        ((74.72, 7.41), (78.27, 7.41)),
        ((78.27, 7.41), (78.27, 8.5)),
        ((80.0, 8.5), (80.0, 6.96)),
        ((80.0, 5.69), (80.0, 3.82)),
        ((80.0, 2.578), (80.0, 1.119)),
        ((80.0, -0.07), (80.0, -0.65)),
        ((80.0, -0.65), (82.0, -0.65)),
        ((80.0, 1.86), (82.0, 1.86)),
        ((80.0, 4.52), (82.0, 4.52)),
        ((80.0, 7.66), (82.0, 7.66)),
        ((70.0, -0.07), (72.43, -0.07)),
        ((74.48, -0.07), (74.72, -0.07)),
        ((74.72, -0.07), (74.72, -1.5)),
    ]

    rects = [
        ((71.73, 8.0), (72.16, 7.41)),
        ((74.72, 8.0), (75.35, 7.41)),
        ((74.72, -0.07), (75.35, -0.7))
    ]


    for (f, t) in lines:
        plt.plot([f[0], t[0]], [f[1], t[1]], **lineargs)

    for (ul, lr) in rects:
        width = lr[0]-ul[0]
        height = lr[1]-ul[1]
        plt.gca().add_patch(
            Rectangle((ul[0], ul[1]), width, height, **rectargs)
        )




