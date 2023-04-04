
name = 'lille'

devs = [
    'dwm1001-1',
    'dwm1001-2',
    'dwm1001-3',
    'dwm1001-10',
    'dwm1001-5',
    'dwm1001-6',
    'dwm1001-7',
    'dwm1001-8',
    'dwm1001-9',
    'dwm1001-4',
    'dwm1001-11',
    'dwm1001-12',
    'dwm1001-13',
    'dwm1001-14'
]

dev_ids = {
    'dwm1001-1': '0x6b93',
    'dwm1001-2': '0xf1c3',
    'dwm1001-3': '0xc240',
    'dwm1001-4': '0x013f',
    'dwm1001-5': '0xb227',
    'dwm1001-6': '0x033b',
    'dwm1001-7': '0xf524',
    'dwm1001-8': '0x37b2',
    'dwm1001-9': '0x15ef',
    'dwm1001-10': '0x7e02',
    'dwm1001-11': '0xad36',
    'dwm1001-12': '0x3598',
    'dwm1001-13': '0x47e0',
    'dwm1001-14': '0x0e92'
}
#
#         0x6b93, // c1 05 0a d2 ca 32
#         0xf1c3, // 2c 8e 38 0e 2c 2f
#         0xc240, // a6 e9 82 fe b9 f2
#         0x013f, // b2 04 f6 33 7c 79
#         0xb227, // e1 bc c5 19 92 66
#         0x033b, // 5e 2a 24 f7 5d 92
#         0xf524, // d8 8d 45 48 12 b9
#         0x37b2, // 12 6a 1d 03 69 47
#         0x15ef, // 2e fb cb ec 8d af
#         0x7e02, // c2 7f f2 eb 6d e0
#         0xad36, // fa 0e a0 b9 19 b7
#         0x3598, // 3c 33 73 36 cb 47
#         0x47e0, // c9 65 a0 9b 5b 9f
#         0x0e92  // 06 79 a6 59 fe 98

dev_positions = {
    'dwm1001-1': (23.31, 0.26, 7.55),
    'dwm1001-2': (24.51, 0.26, 8.96),
    'dwm1001-3': (26.91, 0.26, 8.96),
    'dwm1001-4': (28.11, 0.26, 7.55),
    'dwm1001-5': (25.11, 1.1, 9.51),
    'dwm1001-6': (27.51, 1.1, 9.51),
    'dwm1001-7': (25.11, 3.5, 9.51),
    'dwm1001-8': (27.51, 3.5, 9.51),
    'dwm1001-9': (26.31, 5.9, 9.51),
    'dwm1001-10': (25.11, 7.1, 9.51),
    'dwm1001-11': (27.51, 7.1, 9.51),
    'dwm1001-12': (24.51, 9.39, 7.52),
    'dwm1001-13': (25.71, 9.39, 9.22),
    'dwm1001-14': (26.91, 9.39, 7.52)
}

factory_delays = {
    'dwm1001-1': 16450+8,
    'dwm1001-2': 16450+9,
    'dwm1001-3': 16450+7,
    'dwm1001-4': 16450+22,
    'dwm1001-5': 16450+11,
    'dwm1001-6': 16450+12,
    'dwm1001-7': 16450+7,
    'dwm1001-8': 16450+22,
    'dwm1001-9': 16450+22,
    'dwm1001-10': 16450+22,
    'dwm1001-11': 16450+-14,
    'dwm1001-12': 16450+-9,
    'dwm1001-13': 16450+9,
    'dwm1001-14': 16450+-13
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
            # 1670158055.359572;dwm1001-2;{ "type": "rx", "carrierintegrator": 1434, "rssi": -81, "tx": {"addr": "0xBC48", "sn": 0, "ts": 186691872256}, "rx": [{"addr": "0x471A", "sn": 0, "ts": 185512854420}]}
            log_ts, dev, json_str = line.split(';', 3)
            log_ts = float(log_ts)

            if dev not in dev_set:
                continue

            try:
                msg = json.loads(json_str)
                msg['_log_ts'] = log_ts
                yield (log_ts, dev, msg)
            except json.decoder.JSONDecodeError:
                #print(json_str)
                pass
        except ValueError:
            pass