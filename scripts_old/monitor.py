from typing import List
from serial import Serial
import threading
import time
import sys

def monitor_device(dev_path):

    def dev_print(s):
        print(dev_path + "\t" + s)

    dev_print("Trying to connect...")
    while True:
        try:
            conn = Serial(dev_path, 115200, timeout=1)
            dev_print("Connected!")
            while True:
                line = conn.readline()
                dev_print(line.decode('ascii'))
        except BaseException as e:
            dev_print("Exception:" + str(e))
            time.sleep(1)
            pass






if __name__ == "__main__":
    dev_paths = sys.argv[1:]

    for dev_path in dev_paths:
        t = threading.Thread(target=monitor_device, args=(dev_path,), daemon=True)
        t.start()

    while True:
        time.sleep(1)