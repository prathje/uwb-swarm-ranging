#ifndef MEASUREMENTS_H
#define MEASUREMENTS_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>


typedef float32_t measurement_t;

void estimation_add_measurement(uint8_t a, uint8_t b, measurement_t val);
measurement_t get_mean_measurement(size_t pi);
measurement_t get_var_measurement(size_t pi);

#endif