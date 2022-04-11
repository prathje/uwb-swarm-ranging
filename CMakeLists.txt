cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr)
project(uwb_swarm_ranging)

target_sources(app PRIVATE src/main.c)