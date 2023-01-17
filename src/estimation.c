
#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#include "estimation.h"
#include "node_numbers.h"
#include "uart.h"

#define NUM_NODE_PAIRS ((NUM_NODES*(NUM_NODES-1))/2)
#define NUM_PARAMS (NUM_NODES+NUM_NODE_PAIRS)


LOG_MODULE_REGISTER(estimation);

#if 1
    #define MATRIX_ENTRY_TYPE float16_t
    #define MATRIX arm_matrix_instance_f16
    #define OP_MULT arm_mat_mult_f16
    #define OP_INV arm_mat_inverse_f16
    #define OP_TRANS arm_mat_trans_f16
    #define OP_MEAN arm_mean_f16
#else
    #define MATRIX_ENTRY_TYPE float32_t
    #define MATRIX arm_matrix_instance_f32
    #define OP_MULT arm_mat_mult_f32
    #define OP_INV arm_mat_inverse_f32
    #define OP_TRANS arm_mat_trans_f32
    #define OP_MEAN arm_mean_f32
#endif

#define MAX_NUM_MEASUREMENTS 100

static float32_t measurements[NUM_NODE_PAIRS][MAX_NUM_MEASUREMENTS] = {};
static size_t num_measurements[NUM_NODE_PAIRS] = {0}; // note that this might overflow and wrap around if >= MAX_NUM_MEASUREMENTS


static int32_t actual_antenna_delays[MAX_NUM_NODES] = {
        8, 9, 7, 22, 11, 12, 7, 22, 22, 22, -14, -9, 9, -13
};

static float32_t actual_distances[(MAX_NUM_NODES*(MAX_NUM_NODES-1))/2] = {
        1.8515,
        3.8663, 2.4000,
        4.8000, 3.8663, 1.8515,
        2.7906, 1.1697, 2.0611, 3.6807,
        4.7103, 3.1636, 1.1697, 2.2152, 2.4000,
        4.1928, 3.3407, 3.7470, 4.8311, 2.4000, 3.3941,
        5.6550, 4.4497, 3.3407, 3.8340, 3.3941, 2.4000, 2.4000,
        6.6822, 5.9458, 5.6984, 6.2363, 4.9477, 4.9477, 2.6833, 2.6833,
        7.3394, 6.8883, 7.0942, 7.7219, 6.0000, 6.4622, 3.6000, 4.3267, 1.6971,
        8.2624, 7.4892, 6.8883, 7.1405, 6.4622, 6.0000, 4.3267, 3.6000, 1.6971, 2.4000,
        9.2086, 9.2429, 9.5494, 9.8142, 8.5466, 9.0379, 6.2460, 6.9031, 4.4023, 3.0926, 4.2666,
        9.5868, 9.2122, 9.2122, 9.5868, 8.3167, 8.4881, 5.9276, 6.1657, 3.5531, 2.3850, 2.9271, 2.0809,
        9.8142, 9.5494, 9.2429, 9.2086, 8.7134, 8.5466, 6.4724, 6.2460, 4.0620, 3.5276, 3.0926, 2.4000, 2.0809
};



// TODO: They use up a LOT of memory! We could later allocate them dynamically (if we would need that extra space!)
static MATRIX_ENTRY_TYPE matrix_data_a[(NUM_NODE_PAIRS+NUM_PARAMS) * NUM_PARAMS];
static MATRIX_ENTRY_TYPE matrix_data_b[NUM_PARAMS * (NUM_NODE_PAIRS+NUM_PARAMS)];
static MATRIX_ENTRY_TYPE matrix_data_c[NUM_PARAMS * (NUM_NODE_PAIRS+NUM_PARAMS)];



void print_matrix(MATRIX *m) {
return;
    char buf[32];
    for (int r = 0; r < m->numRows; r++) {
         for (int c = 0; c < m->numCols; c++) {
            int32_t val = m->pData[r*m->numCols + c] * 1000;
            snprintf(buf, sizeof(buf), "%06d ", val);
            uart_out(buf);
        }
        uart_out("\n");
    }
}


inline size_t pair_index(uint8_t a, uint8_t b) {
       if (a > b) {
            size_t num_pairs_before = (a*(a-1))/2; // we have exactly that sum of measurements before
            return num_pairs_before+b;
       } else {
            return pair_index(b, a);
       }
}


void estimation_add_measurement(uint8_t a, uint8_t b, float32_t val) {
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


MATRIX_ENTRY_TYPE get_measurement_for_design_matrix(size_t pi) {
    if (num_measurements[pi] == 0) {
        return 0.0;
    } else {
        float32_t sum = 0.0;

        size_t num = MIN(num_measurements[pi], MAX_NUM_MEASUREMENTS);
        for(size_t i = 0; i < num; i++) {
            sum += measurements[pi][i];
        }

        return sum / ((float)num);
    }
}




void estimate() {

    bool delays_known = true;
    bool distances_known = false;

    // for now we add artificial delay values! -> this means that our matrix should be full rank and we should get a result

    // if 0.0 ->  not known
    float32_t known_delays[NUM_NODES] = {0.0};
    float32_t known_distances[NUM_NODE_PAIRS] = {0.0};


    if (delays_known) {
        for(int i = 0; i < NUM_NODES; i++) {
            known_delays[i] = (float32_t) actual_antenna_delays[i];
            if (known_delays[i] == 0.0) {
                LOG_WRN("Known delay is zero!");
                return;
            }
        }
    }

    if (distances_known) {
        for(int i = 0; i < NUM_NODE_PAIRS; i++) {
            known_distances[i] = actual_distances[i];
            if (known_distances[i] == 0.0) {
                LOG_WRN("Known delay is zero!");
                return;
            }
        }
    }

    LOG_INF("Estimate!!");

    int ret = 0;

    // make sure that matrices are empty! we only need to reset matrix A since the others are overriden
    memset(&matrix_data_a, 0, sizeof(matrix_data_a));

    for(size_t i = 0; i < sizeof(matrix_data_a)/sizeof(matrix_data_a[0]); i++) {
        if (matrix_data_a[i] != 0.0) {
            LOG_WRN("Matrix A not resetted!");
        }
    }

    LOG_DBG("A");


    MATRIX mat_a;
    MATRIX mat_b;
    MATRIX mat_c;

    // assign memory regions
    {
        mat_a.pData = matrix_data_a; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
        mat_b.pData = matrix_data_b; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
        mat_c.pData = matrix_data_c; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));

        // TODO: This cannot happen as we statically defined them.
//        if (mat_a.pData == NULL || mat_b.pData == NULL || mat_c.pData == NULL ) {
//             LOG_INF("Allocation failed...");
//            return false;
//        }
    }

    LOG_DBG("B");
    // Store X in mat_a
    {
        // TODO: we do not these this many rows usually!
        mat_a.numRows = NUM_NODE_PAIRS+NUM_PARAMS;
        mat_a.numCols = NUM_PARAMS;

        // the rows are the measurement values (and known parameters)
        // the columns correspond to the parameters aligned as : Combined Delay Values, Node_Pairs Distances

        // we first store all of the measurements
        for(size_t a = 0; a < NUM_NODES; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }

                size_t pi = pair_index(a, b);

                // TODO: we are currently calculating this measurement twice...
                MATRIX_ENTRY_TYPE val = get_measurement_for_design_matrix(pi);
                if (val != 0.0) {
                    matrix_data_a[(pi * NUM_PARAMS) + a] = 1.0;
                    matrix_data_a[(pi * NUM_PARAMS) + b] = 1.0;
                    matrix_data_a[(pi * NUM_PARAMS) + NUM_NODES + pi] = 2.0;    // the distance is of factor two
                }

                // Y value will be set later
            }
        }




        // we have NODE_PAIRS rows in the matrix now
        // we now store the known delay values (up to NUM_NODES)
        for(size_t a = 0; a < NUM_NODES; a++) {

            if (known_delays[a] != 0.0) {
                matrix_data_a[NUM_NODE_PAIRS * NUM_PARAMS + a * NUM_PARAMS + a] = 1.0;
            }
            // Y value will be set later
        }

         // we have NODE_PAIRS+NUM_NODES rows in the matrix now
        // we now store the known distance values (NODE_PAIRS)
        for(size_t a = 0; a < NUM_NODES; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);

                if (known_distances[pi] != 0.0) {
                    matrix_data_a[(NUM_NODE_PAIRS+NUM_NODES) * NUM_PARAMS + pi * NUM_PARAMS + NUM_NODES + pi] = 1.0;
                }
                // Y value will be set later
            }
        }

        LOG_DBG("Matrix X:");
        print_matrix(&mat_a);
    }

    // transpose X to mat_b
    {
        mat_b.numRows = NUM_PARAMS;
        mat_b.numCols = NUM_NODE_PAIRS+NUM_PARAMS;

        ret = OP_TRANS(&mat_a, &mat_b);
         if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... A");
            return false;
        }
    }


    // multiply X^T (mat_b) with X (mat_a) and store in mat_c
    {
        mat_c.numRows = NUM_PARAMS;
        mat_c.numCols = NUM_PARAMS;

        ret = OP_MULT(&mat_b, &mat_a, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... B");
            return false;
        }
    }

    // invert (X^T X) (mat_c) and store in mat_a
    // we keep mat_b as we need the transpose in the next step
    {
        mat_a.numRows = NUM_PARAMS;
        mat_a.numCols = NUM_PARAMS;

        ret = OP_INV(&mat_c, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... C");
            return false;
        }
    }

    // multiply (X^T X)^{-1} (mat_a) with X^T (mat_b) and store in mat_c
    {
        mat_c.numRows = NUM_PARAMS;
        mat_c.numCols = NUM_PARAMS+NUM_NODE_PAIRS;

        ret = OP_MULT(&mat_a, &mat_b, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... D");
            return false;
        }
    }


    // load Y into mat_b
    {
        mat_b.numRows = NUM_NODE_PAIRS+NUM_PARAMS;
        mat_b.numCols = 1;
        //TODO: implement this

        // we first store all of the measurements
        for(size_t a = 0; a < NUM_NODES; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);
                MATRIX_ENTRY_TYPE val = get_measurement_for_design_matrix(pi);
                if (val != 0.0) {
                    matrix_data_b[pi] = val * 2.0; // we need to scale this accordingly
                }
            }
        }

        // we have NODE_PAIRS rows in the matrix now
        // we now store the known delay values (up to NUM_NODES)
        for(size_t a = 0; a < NUM_NODES; a++) {

            if (known_delays[a] != 0.0) {
                matrix_data_b[NUM_NODE_PAIRS + a] = known_delays[a];
            }
            // Y value will be set later
        }

         // we have NODE_PAIRS+NUM_NODES rows in the matrix now
        // we now store the known distance values (NODE_PAIRS)
        for(size_t a = 0; a < NUM_NODES; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);

                if (known_distances[pi] != 0.0) {
                    matrix_data_b[(NUM_NODE_PAIRS+NUM_NODES) + pi] = known_distances[pi];
                }
            }
        }
        LOG_DBG("Matrix Y:");
        print_matrix(&mat_b);
    }

    // multiply (X^T X)^{-1}  X^T (mat_c) with Y (mat_b) and store in mat_a
    {
        mat_a.numRows = NUM_PARAMS;
        mat_a.numCols = 1;

        ret = OP_MULT(&mat_c, &mat_b, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... E");
            return false;
        }
    }

    LOG_DBG("Estimate:");
    print_matrix(&mat_a);

    for(size_t a = 0; a < NUM_NODES; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }

                size_t pi = pair_index(a, b);

                MATRIX_ENTRY_TYPE tof_in_uwb_us = matrix_data_a[NUM_NODES + pi];

                float est_distance_in_m = tof_in_uwb_us*SPEED_OF_LIGHT_M_PER_UWB_US;
                int est_cm = est_distance_in_m*100;
                LOG_DBG("Estimated cm %d, %d: %d", a, b, est_cm);
            }
        }

    return true;
}


//
//bool matrix_inference() {
//    LOG_INF("matrix_test_f16");
//
//    int ret = 0;
//
//    // TODO: They use up a LOT of memory!
//    static MATRIX_ENTRY_TYPE matrix_data_a[NUM_PARAMS * NUM_PARAMS];
//    static MATRIX_ENTRY_TYPE matrix_data_b[NUM_PARAMS * NUM_PARAMS];
//    static MATRIX_ENTRY_TYPE matrix_data_c[NUM_PARAMS * NUM_PARAMS];
//
//    MATRIX mat_a;
//    MATRIX mat_b;
//    MATRIX mat_c;
//
//    // assign memory regions
//    {
//        mat_a.pData = matrix_data_a; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
//        mat_b.pData = matrix_data_b; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
//        mat_c.pData = matrix_data_c; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
//
//
//        if (mat_a.pData == NULL || mat_b.pData == NULL || mat_c.pData == NULL ) {
//             LOG_INF("Allocation failed...");
//            return false;
//        }
//    }
//
//
//    // Store X in mat_a
//    {
//        // TODO: actually store it!
//        mat_a.numRows = NUM_PARAMS;
//        mat_a.numCols = NUM_PARAMS;
//    }
//
//    // transpose X to mat_b
//    {
//        mat_b.numRows = NUM_PARAMS;
//        mat_b.numCols = NUM_PARAMS;
//
//        ret = OP_TRANS(&mat_a, &mat_b);
//         if (ret == ARM_MATH_SIZE_MISMATCH ) {
//             LOG_INF("SIZE MISMATCH... A");
//            return false;
//        }
//    }
//
//    // multiply X^T (mat_b) with X (mat_a) and store in mat_c
//    {
//        mat_c.numRows = NUM_PARAMS;
//        mat_c.numCols = NUM_PARAMS;
//
//        ret = OP_MULT(&mat_b, &mat_a, &mat_c);
//
//        if (ret == ARM_MATH_SIZE_MISMATCH ) {
//             LOG_INF("SIZE MISMATCH... B");
//            return false;
//        }
//    }
//
//
//    // invert (X^T X) (mat_c) and store in mat_a
//    // we keep mat_b as we need the transpose in the next step
//    {
//        mat_a.numRows = NUM_PARAMS;
//        mat_a.numCols = NUM_PARAMS;
//
//        ret = OP_MULT(&mat_c, &mat_a);
//
//        if (ret == ARM_MATH_SIZE_MISMATCH ) {
//             LOG_INF("SIZE MISMATCH... C");
//            return false;
//        }
//    }
//
//    // multiply (X^T X)^{-1} (mat_a) with X^T (mat_b) and store in mat_c
//    {
//        mat_c.numRows = NUM_PARAMS;
//        mat_c.numCols = NUM_PARAMS;
//
//        ret = OP_MULT(&mat_a, &mat_b, &mat_c);
//
//        if (ret == ARM_MATH_SIZE_MISMATCH ) {
//             LOG_INF("SIZE MISMATCH... D");
//            return false;
//        }
//    }
//
//
//    // load Y into mat_b
//    {
//        //TODO: implement this
//        mat_b.numRows = NUM_PARAMS;
//        mat_b.numCols = 1;
//    }
//
//    // multiply (X^T X)^{-1}  X^T (mat_c) with Y (mat_b) and store in mat_a
//    {
//        mat_a.numRows = NUM_PARAMS;
//        mat_a.numCols = 1;
//
//        ret = OP_MULT(&mat_c, &mat_b, &mat_a);
//
//        if (ret == ARM_MATH_SIZE_MISMATCH ) {
//             LOG_INF("SIZE MISMATCH... E");
//            return false;
//        }
//    }
//     LOG_INF("Done!!");
//
//    return true;
//}