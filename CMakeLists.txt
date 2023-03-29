cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr)
project(uwb_swarm_ranging)

target_sources(app PRIVATE src/main.c src/testbed.c src/nodes.c src/measurements.c src/estimation.c src/uart.c src/history.c)