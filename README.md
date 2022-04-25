# UWB Swarm Ranging

This project contains the PoC for Ultra-Wideband ranging using a single, broadcast message for simultaneous rangings to all other devices.


## Setup
Setup Zephyr and West as usual. As the Decawave Driver does not allow precise timings, we added some overrides which need to be manually applied (see override directory).

Tested based on commit 6d56b829423056819c4baaafd6c66957752e22f8, while commit eeb4434d2eb5f2c978c59a439688c1f3f46e8bf8 has been reverted due to scheduling exceptions (already included in the overrides).

## Build

Build the project for the Decawave DWM1001 module:
```bash
west build -b decawave_dwm1001_dev --pristine auto
```

You can then flash the boards one by one:
```bash
west flash
```


## Scripts

### Serial Monitor

Use the monitor Python script to display JSON outputs of also multiple devices.
```
pip3 install pyserial
python3 monitor.py </dev/tty.dev1> </dev/tty.dev2>
```

The result should look something like this:
![Example](img/example.png)
