#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

void estimation_add_measurement(uint8_t a, uint8_t b, float32_t val);
float32_t get_mean_measurement(size_t pi);

#endif