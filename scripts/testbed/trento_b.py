
name = 'trento_b'

devs = [
    'dwm1001.160',
    'dwm1001.161',
    'dwm1001.162',
    'dwm1001.163',
    'dwm1001.164',
    'dwm1001.165',
    'dwm1001.166',
    'dwm1001.167',
    'dwm1001.168',
    'dwm1001.169',
    'dwm1001.170',
    'dwm1001.171',
    'dwm1001.172',
    'dwm1001.173'
]

dev_ids = {
    'dwm1001.160': '0x330b',
    'dwm1001.161': '0xbee3',
    'dwm1001.162': '0x17e3',
    'dwm1001.163': '0x427e',
    'dwm1001.164': '0xdd4f',
    'dwm1001.165': '0x292b',
    'dwm1001.166': '0x69f6',
    'dwm1001.167': '0x3843',
    'dwm1001.168': '0x3aea',
    'dwm1001.169': '0x2df2',
    'dwm1001.170': '0x3392',
    'dwm1001.171': '0x5418',
    'dwm1001.172': '0x869a',
    'dwm1001.173': '0x1582'
}

dev_positions = {
    'dwm1001.160': (130.29, 4.7, 0.0),
    'dwm1001.161': (130.29, 6.56, 0.0),
    'dwm1001.162': (126.95, 6.84, 0.0),
    'dwm1001.163': (124.52, 6.82, 0.0),
    'dwm1001.164': (122.79, 6.8, 0.0),
    'dwm1001.165': (120.99, 6.78, 0.0),
    'dwm1001.166': (120.07, 4.64, 0.0),
    'dwm1001.167': (120.07, 2.28, 0.0),
    'dwm1001.168': (120.98, 0.19, 0.0),
    'dwm1001.169': (122.76, 0.2, 0.0),
    'dwm1001.170': (124.62, 0.19, 0.0),
    'dwm1001.171': (127.05, 0.18, 0.0),
    'dwm1001.172': (129.99, 0.19, 0.0),
    'dwm1001.173': (130.27, 2.31, 0.0)
}

factory_delays = {
    'dwm1001.160': 16458,
    'dwm1001.161': 16460,
    'dwm1001.162': 16472,
    'dwm1001.163': 16481,
    'dwm1001.164': 16459,
    'dwm1001.165': 16472,
    'dwm1001.166': 16460,
    'dwm1001.167': 16472,
    'dwm1001.168': 16461,
    'dwm1001.169': 16472,
    'dwm1001.170': 16458,
    'dwm1001.171': 16472,
    'dwm1001.172': 16472,
    'dwm1001.173': 16472
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
                print(json_str)
                pass
        except ValueError:
            pass