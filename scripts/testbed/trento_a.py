

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