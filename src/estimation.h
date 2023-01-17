#ifndef ESTIMATION_H
#define ESTIMATION_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#define SPEED_OF_LIGHT_M_PER_S 299792458.0
#define SPEED_OF_LIGHT_M_PER_UWB_US ((SPEED_OF_LIGHT_M_PER_S * 1.0E-15) * 15650.0) // around 0.00469175196

void estimate();


size_t pair_index(uint8_t a, uint8_t b);

void estimation_add_measurement(uint8_t a, uint8_t b, float32_t val);

#endif