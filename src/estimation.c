
#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <logging/log.h>
#include <stdio.h>

#include "estimation.h"
#include "nodes.h"
#include "uart.h"
#include "measurements.h"

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



// the maximum of supported nodes for estimation (limited by memory sadly)
#define EST_MAX_NODES 8
#define EST_MAX_PAIRS (PAIRS(EST_MAX_NODES))
#define EST_MAX_PARAMS (EST_MAX_NODES + EST_MAX_PAIRS)
#define EST_MAX_INPUTS (EST_MAX_PAIRS + EST_MAX_PARAMS)

// TODO: They use up a LOT of memory! We could later allocate them dynamically (if we would need that extra space!)
static MATRIX_ENTRY_TYPE matrix_data_a[EST_MAX_INPUTS * EST_MAX_PARAMS];
static MATRIX_ENTRY_TYPE matrix_data_b[EST_MAX_PARAMS * EST_MAX_INPUTS];
static MATRIX_ENTRY_TYPE matrix_data_c[EST_MAX_PARAMS * EST_MAX_INPUTS];

void print_matrix(MATRIX *m) {
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


static void estimate(
    uint8_t num_nodes,
    float32_t delays_out[EST_MAX_NODES],
    float32_t tofs_out[EST_MAX_PAIRS],
    float32_t mean_measurements[EST_MAX_PAIRS],
    bool delay_known[EST_MAX_NODES],
    float32_t known_delays[EST_MAX_NODES],
    bool tof_known[EST_MAX_PAIRS],
    float32_t known_tofs[EST_MAX_PAIRS]
) {

    if (num_nodes >= EST_MAX_NODES) {
        LOG_ERR("Num Nodes too high!");
        return;
    }

    int ret = 0;

    uint32_t num_pairs = PAIRS(num_nodes); // we use the actual number of input pairs
    uint32_t num_params = num_nodes+num_pairs; // we use the actual number of input pairs

    uint32_t num_input_rows = PAIRS(num_nodes); // we use the actual number of input pairs
    // we add all known delays and distances as input rows as well!
    for(int i = 0; i < num_nodes; i++) {
        if (delay_known[i]) {
            num_input_rows++;
        }
    }
     for(int i = 0; i < num_nodes; i++) {
        if (tof_known[i]) {
            num_input_rows++;
        }
    }

    // make sure that matrices are empty! we only need to reset matrix A since the others are overriden anyway
    // this zero matrix a
    memset(&matrix_data_a, 0, sizeof(matrix_data_a));

    MATRIX mat_a;
    MATRIX mat_b;
    MATRIX mat_c;

    // assign memory regions
    {
        mat_a.pData = matrix_data_a; //malloc(EST_MAX_PARAMS * EST_MAX_PARAMS * sizeof(float16_t));
        mat_b.pData = matrix_data_b; //malloc(EST_MAX_PARAMS * EST_MAX_PARAMS * sizeof(float16_t));
        mat_c.pData = matrix_data_c; //malloc(EST_MAX_PARAMS * EST_MAX_PARAMS * sizeof(float16_t));

        // TODO: This cannot happen as we statically defined them.
//        if (mat_a.pData == NULL || mat_b.pData == NULL || mat_c.pData == NULL ) {
//             LOG_INF("Allocation failed...");
//            return false;
//        }
    }

    // Store X in mat_a
    {
        // TODO: we do not these this many rows usually!
        mat_a.numRows = num_input_rows;
        mat_a.numCols = num_params;

        // the rows are the measurement values (and known parameters)
        // the columns correspond to the parameters aligned as : Combined Delay Values, Node_Pairs Distances

        // we first store all of the measurements
        for(size_t a = 0; a < num_nodes; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }

                size_t pi = pair_index(a, b);

                // TODO: we are currently calculating this measurement twice...
                MATRIX_ENTRY_TYPE val = (MATRIX_ENTRY_TYPE)get_mean_measurement(pi);
                if (val != 0.0) {
                    matrix_data_a[(pi * num_params) + a] = 1.0;
                    matrix_data_a[(pi * num_params) + b] = 1.0;
                    matrix_data_a[(pi * num_params) + num_nodes + pi] = 2.0;    // the distance is of factor two
                }

                // Y value will be set later
            }
        }

        // we have num_pairs rows in the matrix now
        // we now store the known delay values (up to num_nodes)
        int input_row = num_pairs;

        for(size_t a = 0; a < num_nodes; a++) {
            if (delay_known[a]) {
                matrix_data_a[input_row * num_params + a] = 1.0;
                input_row++; // increase the row
            }
            // Y value will be set later
        }

         // we have NODE_PAIRS+num_nodes rows in the matrix now
        // we now store the known distance values (NODE_PAIRS)
        for(size_t a = 0; a < num_nodes; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);

                if (tof_known[pi]) {
                    matrix_data_a[input_row * num_params + num_nodes + pi] = 1.0;
                    input_row++; // increase the row
                }
                // Y value will be set later
            }
        }

        LOG_DBG("Matrix X:");
        print_matrix(&mat_a);
    }

    // transpose X to mat_b
    {
        mat_b.numRows = num_params;
        mat_b.numCols = num_input_rows;

        ret = OP_TRANS(&mat_a, &mat_b);
         if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... A");
            return false;
        }
    }


    // multiply X^T (mat_b) with X (mat_a) and store in mat_c
    {
        mat_c.numRows = num_params;
        mat_c.numCols = num_params;

        ret = OP_MULT(&mat_b, &mat_a, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... B");
            return false;
        }
    }

    // invert (X^T X) (mat_c) and store in mat_a
    // we keep mat_b as we need the transpose in the next step
    {
        mat_a.numRows = num_params;
        mat_a.numCols = num_params;

        ret = OP_INV(&mat_c, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... C");
            return false;
        }
    }

    // multiply (X^T X)^{-1} (mat_a) with X^T (mat_b) and store in mat_c
    {
        mat_c.numRows = num_params;
        mat_c.numCols = num_input_rows;

        ret = OP_MULT(&mat_a, &mat_b, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... D");
            return false;
        }
    }


    // load Y into mat_b
    {
        mat_b.numRows = num_input_rows;
        mat_b.numCols = 1;
        //TODO: implement this

        // we first store all of the measurements
        for(size_t a = 0; a < num_nodes; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);
                MATRIX_ENTRY_TYPE val = get_mean_measurement(pi);
                if (val != 0.0) {
                    matrix_data_b[pi] = val * 2.0; // we need to scale this accordingly
                }
            }
        }

        // we have num_pairs rows in the matrix now
        // we now store the known delay values (up to num_nodes)
        int input_row = num_pairs;

        for(size_t a = 0; a < num_nodes; a++) {
            if (delay_known[a]) {
                matrix_data_b[input_row] = known_delays[a];
                input_row++; // increase the row
            }
        }

        // we have NODE_PAIRS+num_nodes rows in the matrix now
        // we now store the known distance values (NODE_PAIRS)
        for(size_t a = 0; a < num_nodes; a++) {
            for(size_t b = 0; b < a; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);

                if (tof_known[pi]) {
                    matrix_data_b[input_row] = known_tofs[pi];
                    input_row++; // increase the row
                }
            }
        }

        LOG_DBG("Matrix Y:");
        print_matrix(&mat_b);
    }

    // multiply (X^T X)^{-1}  X^T (mat_c) with Y (mat_b) and store in mat_a
    {
        mat_a.numRows = num_params;
        mat_a.numCols = 1;

        ret = OP_MULT(&mat_c, &mat_b, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... E");
            return;
        }
    }

    // copy the antenna delays
    for (int i = 0; i < num_nodes; i++) {
        delays_out[i] = matrix_data_a[i];
    }

    // copy the distances

    for(int p = 0; p < num_pairs; p++) {
        tofs_out[p] = matrix_data_a[num_nodes + p];
    }

    LOG_DBG("Estimate:");
    print_matrix(&mat_a);

}


void estimate_all() {

    static float32_t delays_out[EST_MAX_NODES] = {0.0};
    static float32_t tofs_out[EST_MAX_PAIRS] = {0.0};
    static float32_t mean_measurements[EST_MAX_PAIRS] = {0.0};

    static bool delay_known[EST_MAX_NODES];
    static float32_t known_delays[EST_MAX_NODES];
    static bool tof_known[EST_MAX_PAIRS];
    static float32_t known_tofs[EST_MAX_PAIRS];

    int64_t estimate_start = 0;
    int64_t estimate_duration = 0;


    for(int n = 2; n <= 3; n++) {

        // TODO: Should we reorder nodes?

        //initialize known values (measurements, delays, distances
        for(int i = 0; i < n; i++) {
            known_delays[i] = (float32_t) node_factory_antenna_delays[i];
        }

         for(int p = 0; p < PAIRS(n); p++) {
            // TODO: If we reorder, we have to be careful with those distances!
            mean_measurements[p] = get_mean_measurement(p);
            known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_US;
        }

        // first estimate antenna delays based on all known distances
        {
            memset(tof_known, 1, sizeof(tof_known));
            memset(delay_known, 0, sizeof(delay_known));

            estimate_start = k_uptime_get();
            estimate(
                n,
                delays_out,
                tofs_out,
                mean_measurements,
                delay_known,
                known_delays,
                tof_known,
                known_tofs
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            LOG_INF("Estimation required: ms: %lld", estimate_duration);

            // log the antenna delays
            for (int i = 0; i < n; i++) {
                int est = (int)delays_out[i];
                LOG_DBG("Estimated antenna delay %d, %d (%d)", i, est, node_factory_antenna_delays[i]);
            }
             // log the distances

              for(size_t a = 0; a < n; a++) {
                for(size_t b = 0; b < a; b++) {
                    if (a == b) {
                        continue;
                    }

                    size_t pi = pair_index(a, b);
                    float32_t est = tofs_out[pi];
                    float32_t est_distance_in_m = est*SPEED_OF_LIGHT_M_PER_UWB_US;
                    int est_cm = est_distance_in_m*100;
                    int actual_cm = known_tofs[pi]*100;
                    LOG_DBG("Estimated cm %d, %d: %d (%d)", a, b, est_cm, actual_cm);
                }
            }
        }
    }
}