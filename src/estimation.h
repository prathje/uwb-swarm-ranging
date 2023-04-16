#ifndef ESTIMATION_H
#define ESTIMATION_H

#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#define EST_MAX_NODES (NUM_NODES)
#define EST_MAX_PAIRS (PAIRS(EST_MAX_NODES))
#define EST_MAX_PARAMS (EST_MAX_NODES)
#define EST_MAX_INPUTS (EST_MAX_PAIRS)

//#define SPEED_OF_LIGHT_M_PER_S 299792458.0
#define SPEED_OF_LIGHT_M_PER_S 299702547.236
#define SPEED_OF_LIGHT_M_PER_UWB_TU ((SPEED_OF_LIGHT_M_PER_S * 1.0E-15) * 15650.0) // around 0.00469175196

void estimate_all(uint16_t round);

#endif