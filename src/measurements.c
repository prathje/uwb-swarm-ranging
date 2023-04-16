#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <stdio.h>
#include <logging/log.h>

#include "measurements.h"
#include "nodes.h"


LOG_MODULE_REGISTER(measurements);

#if 1

static struct {
    size_t count;
    float32_t mean;
    float32_t M2;
} aggr[NUM_PAIRS];

// For a new value newValue, compute the new count, new mean, the new M2.
// mean accumulates the mean of the entire dataset
// M2 aggregates the squared distance from the mean
// count aggregates the number of samples seen so far
void aggr_add(int index, float32_t newValue) {
    aggr[index].count += 1;
    float32_t delta = newValue - aggr[index].mean;
    aggr[index].mean += delta / (float32_t) aggr[index].count;
    float32_t delta2 = newValue - aggr[index].mean;
    aggr[index].M2 += delta * delta2;
}

void aggr_reset(int index) {
    aggr[index].count = 0;
    aggr[index].mean = 0.0;
    aggr[index].M2 = 0.0;
}

float32_t aggr_mean(int index) {
    return aggr[index].mean;
}

float32_t aggr_variance(int index) {
    return aggr[index].M2  / (float32_t) (aggr[index].count);
}

void estimation_add_measurement(uint8_t a, uint8_t b, measurement_t val) {
    if (a == b) {
        LOG_WRN("Tried to add a measurement to itself!");
        return;
    } else if (val == 0.0) {
        LOG_WRN("Tried to add a zero measurement!");
        return;
    }

    size_t pi = pair_index(a, b);
    aggr_add(pi, val);
}

measurement_t get_mean_measurement(size_t pi) {
    return aggr_mean(pi);
}

measurement_t get_var_measurement(size_t pi) {
    return aggr_variance(pi);
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