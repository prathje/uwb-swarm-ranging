
#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <logging/log.h>

#include "estimation.h"
#include "node_numbers.h"

#define NUM_NODE_PAIRS ((NUM_NODES*(NUM_NODES-1))/2)
#define NUM_PARAMS (NUM_NODES+NUM_NODE_PAIRS)


LOG_MODULE_REGISTER(estimation);

#if 0
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

static float32_t measurements[NUM_NODE_PAIRS][MAX_NUM_MEASUREMENTS] = {0.0};
static size_t num_measurements[NUM_NODE_PAIRS] = {0}; // note that this might overflow and wrap around if >= MAX_NUM_MEASUREMENTS

static float32_t known_delays[NUM_NODES] = {0.0};          // if 0.0 ->  delay not known
static float32_t known_distances[NUM_NODE_PAIRS] = {0.0};  // if 0.0 ->  distance now known

// TODO: They use up a LOT of memory! We could later allocate them dynamically (if we would need that extra space!)


static MATRIX_ENTRY_TYPE matrix_data_a[(NUM_NODE_PAIRS+NUM_PARAMS) * NUM_PARAMS];
static MATRIX_ENTRY_TYPE matrix_data_b[NUM_PARAMS * (NUM_NODE_PAIRS+NUM_PARAMS)];
static MATRIX_ENTRY_TYPE matrix_data_c[NUM_PARAMS * (NUM_NODE_PAIRS+NUM_PARAMS)];



void print_matrix(MATRIX *m) {


    for (int r = 0; r < m->numRows; r++) {
         for (int c = 0; c < m->numCols; c++) {


        }
    }
}


size_t pair_index(uint8_t a, uint8_t b) {
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
                    matrix_data_a[(pi * NUM_PARAMS) + NUM_NODES + pi] = 1.0;
                }

                // Y value will be set later
            }
        }

        LOG_DBG("B2");

        // we have NODE_PAIRS rows in the matrix now
        // we now store the known delay values (up to NUM_NODES)
        for(size_t a = 0; a < NUM_NODES; a++) {

            if (known_delays[a] != 0.0) {
                matrix_data_a[NUM_NODE_PAIRS * NUM_PARAMS + a * NUM_PARAMS + a] = 1.0;
            }
            // Y value will be set later
        }

        LOG_DBG("B3");
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
    }

    LOG_DBG("C");
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

    LOG_DBG("D");

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

    LOG_DBG("E");
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
     LOG_INF("Done!!");

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