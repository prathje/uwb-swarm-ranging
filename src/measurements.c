#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <stdio.h>
#include <logging/log.h>

#include "measurements.h"
#include "nodes.h"


LOG_MODULE_REGISTER(measurements);

#if 0
#define MAX_NUM_MEASUREMENTS 1000
// note that we collect measurements of all nodes for now
static float32_t measurements[NUM_PAIRS][MAX_NUM_MEASUREMENTS] = {};
static size_t num_measurements[NUM_PAIRS] = {0}; // note that this might overflow and wrap around if >= MAX_NUM_MEASUREMENTS

void estimation_add_measurement(uint8_t a, uint8_t b, measurement_t val) {
    if (a == b) {
        LOG_WRN("Tried to add a measurement to itself!");
        return;
    } else if (val == 0.0) {
        LOG_WRN("Tried to add a zero measurement!");
        return;
    }

    size_t pi = pair_index(a, b);
    measurements[pi][num_measurements[pi] % MAX_NUM_MEASUREMENTS] = val;    // we save it in a circular buffer
    num_measurements[pi]++;
}

measurement_t get_mean_measurement(size_t pi) {
    if (num_measurements[pi] == 0) {
        return 0.0;
    } else {
        float32_t sum = 0.0;

        size_t num = MIN(num_measurements[pi], MAX_NUM_MEASUREMENTS);
        for(size_t i = 0; i < num; i++) {
            sum += measurements[pi][i];
        }

        return sum / ((float32_t)num);
    }
}

#else

// note that we collect measurements of all nodes for now
static float32_t measurements_sum[NUM_PAIRS] = {0.0};
static float32_t measurements_sum_of_squares[NUM_PAIRS] = {0.0};

static size_t num_measurements[NUM_PAIRS] = {0}; // note that this might overflow and wrap around if >= MAX_NUM_MEASUREMENTS

void estimation_add_measurement(uint8_t a, uint8_t b, measurement_t val) {
    if (a == b) {
        //LOG_WRN("Tried to add a measurement to itself!");
        return;
    } else if (val == 0.0) {
        //LOG_WRN("Tried to add a zero measurement!");
        return;
    }

    size_t pi = pair_index(a, b);

    measurements_sum[pi] += val;    // we save it in a circular buffer
    measurements_sum_of_squares[pi] += val*val;    // we save it in a circular buffer
    num_measurements[pi]++; // we also save the amount of measurements
}

measurement_t get_mean_measurement(size_t pi) {
    if (num_measurements[pi] == 0) {
        return 0.0;
    } else {
        return measurements_sum[pi] / ((float32_t)num_measurements[pi]);
    }
}

#endif