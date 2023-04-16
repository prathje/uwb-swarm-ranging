
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
    typedef float16_t matrix_entry_t;
    #define MATRIX arm_matrix_instance_f16
    #define OP_MULT arm_mat_mult_f16
    #define OP_INV arm_mat_inverse_f16
    #define OP_TRANS arm_mat_trans_f16
    #define OP_MEAN arm_mean_f16
#else
    typedef float32_t matrix_entry_t;
    #define MATRIX arm_matrix_instance_f32
    #define OP_MULT arm_mat_mult_f32
    #define OP_INV arm_mat_inverse_f32
    #define OP_TRANS arm_mat_trans_f32
    #define OP_MEAN arm_mean_f32
#endif



#define MAX_SD_IN_M (0.1)
#define MAX_SD_IN_UWB_TS (MAX_SD_IN_M/SPEED_OF_LIGHT_M_PER_UWB_TU)


#define MAX_VAR_IN_M (MAX_SD_IN_UWB_TS*MAX_SD_IN_UWB_TS)




// the maximum of supported nodes for estimation (limited by memory sadly)
// TODO: They use up a LOT of memory! We could later allocate them dynamically (if we would need that extra space!)

void print_matrix(MATRIX *m) {
return;
    char buf[32];
    for (int r = 0; r < m->numRows; r++) {
         for (int c = 0; c < m->numCols; c++) {
            int32_t val = m->pData[r*m->numCols + c];
            snprintf(buf, sizeof(buf), "%d ", val);
            uart_out(buf);
        }
        uart_out("\n");
    }
}


static matrix_entry_t matrix_data_a[EST_MAX_INPUTS * EST_MAX_INPUTS];
static matrix_entry_t matrix_data_b[EST_MAX_PARAMS * EST_MAX_INPUTS];
static matrix_entry_t matrix_data_c[EST_MAX_INPUTS * EST_MAX_INPUTS];

static void estimate(
    uint8_t num_nodes,
    float32_t delays_out[EST_MAX_NODES],
    float32_t tofs_out[EST_MAX_PAIRS],
    float32_t mean_measurements[EST_MAX_PAIRS],
    float32_t var_measurements[EST_MAX_PAIRS],
    bool delay_known[EST_MAX_NODES],
    float32_t known_delays[EST_MAX_NODES],
    bool tof_known[EST_MAX_PAIRS],
    float32_t known_tofs[EST_MAX_PAIRS],
    float32_t max_pair_variance
) {

    if (num_nodes > EST_MAX_NODES) {
        LOG_ERR("Num Nodes too high!");
        return;
    }

    int ret = 0;

    uint32_t num_pairs = PAIRS(num_nodes); // we use the actual number of input pairs
    uint32_t num_input_rows = PAIRS(num_nodes); // we use the actual number of input pairs
    uint32_t num_input_cols = 0;

    // we add all unknown delays and distances as columns
    for(int i = 0; i < num_nodes; i++) {
        if (!delay_known[i]) {
            num_input_cols++;
        }
    }

    if (num_input_cols > EST_MAX_NODES) {
        LOG_ERR("Inference of TOFs currently not supported! Only antenna delays as of now");
        return;
    }


    for(int i = 0; i < num_pairs; i++) {
        if (!tof_known[i]) {
            num_input_cols++;
        }
        // TDDO: We could also filter by variance here
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
        mat_a.numCols = num_input_cols;

        // as the parameter indices

        int param_index = 0;

        // first add all the delays
        for(int a = 0; a < num_nodes; a++) {
            if (delay_known[a]) {
                continue; // this one is not relevant here
            }

            // we have to loop through all nodes to get also the reverse combination
            for(size_t b = 0; b < num_nodes; b++) {
                if (a == b) {
                    continue;
                }
                size_t pi = pair_index(a, b);
                if (max_pair_variance > 0.0 && var_measurements != NULL && var_measurements[pi] >= max_pair_variance) {
                    continue; // we skip this entry as it has too high variance, note that we do not change parameter indices because of it
                }
                matrix_data_a[(pi * num_input_cols) + param_index] = 1.0;
            }
            param_index++;
        }

         for(int a = 0; a < num_nodes; a++) {
            for(int b = 0; b < a; b++) {
                size_t pi = pair_index(a, b);

                 if (tof_known[pi]) {
                     continue; // this one is not relevant here
                 }

                 matrix_data_a[(pi * num_input_cols) + param_index] = 2.0; // the distance is of factor two
                 param_index++;
            }
         }

         if (param_index != num_input_cols) {
            LOG_ERR("aram_index != num_input_cols, %d, %d", param_index, num_input_cols);
            return;
         }

        //LOG_DBG("Matrix X:");
        //print_matrix(&mat_a);
    }

    // transpose X to mat_b
    {
        mat_b.numRows = num_input_cols;
        mat_b.numCols = num_input_rows;

        ret = OP_TRANS(&mat_a, &mat_b);
         if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... A");
            return false;
        }
    }


    // multiply X^T (mat_b) with X (mat_a) and store in mat_c
    {
        mat_c.numRows = num_input_cols;
        mat_c.numCols = num_input_cols;

        ret = OP_MULT(&mat_b, &mat_a, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... B");
            return false;
        }
    }

    // invert (X^T X) (mat_c) and store in mat_a
    // we keep mat_b as we need the transpose in the next step
    {
        mat_a.numRows = num_input_cols;
        mat_a.numCols = num_input_cols;

        ret = OP_INV(&mat_c, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... C");
            return false;
        }
    }

    // multiply (X^T X)^{-1} (mat_a) with X^T (mat_b) and store in mat_c
    {
        mat_c.numRows = num_input_cols;
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
                size_t pi = pair_index(a, b);
                float32_t val;
                if (delay_known[a] && delay_known[b] && tof_known[pi]) {
                    val = 0.0; // nothing todo here! this is a null row...
                }
                else if (delay_known[a] && max_pair_variance > 0.0 && var_measurements != NULL && var_measurements[pi] >= max_pair_variance) {
                    val = 0.0; // we remove this entire row BUT just for antenna delay estimation!
                }
                else {
                    val = mean_measurements[pi] * 2.0;  // we need to scale this accordingly

                    if (delay_known[a]) {
                        val -= known_delays[a];
                    }
                    if (delay_known[b]) {
                        val -= known_delays[b];
                    }
                    if(tof_known[pi]) {
                        val -= known_tofs[pi]*2.0; //tof has factor two!
                    }

                }
                matrix_data_b[pi] = val;
            }
        }

//        LOG_DBG("Matrix Y:");
//        print_matrix(&mat_b);
    }

    // multiply (X^T X)^{-1}  X^T (mat_c) with Y (mat_b) and store in mat_a
    {
        mat_a.numRows = num_input_cols;
        mat_a.numCols = 1;

        ret = OP_MULT(&mat_c, &mat_b, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... E");
            return;
        }
    }


//    LOG_DBG("Estimate:");
//    print_matrix(&mat_a);

    {
        // copy stuff to out buffers
        int param_index = 0;

        // first copy all the delays
        for(int a = 0; a < num_nodes; a++) {
            if (delay_known[a]) {
                if (delays_out != NULL){
                    delays_out[a] = known_delays[a];
                }
            } else {
                if (delays_out != NULL){
                    delays_out[a] = matrix_data_a[param_index];
                }
                param_index++;
            }
        }


        // TODO: we cannot infer tofs because of memory constraints...
        // so we simply calculate them manually here...

        for(int a = 0; a < num_nodes; a++) {
            for(int b = 0; b < a; b++) {
                size_t pi = pair_index(a, b);

                 if (tof_known[pi]) {
                    if (tofs_out != NULL) {
                        tofs_out[pi] = known_tofs[pi];
                    }
                 } else {
                    if (tofs_out != NULL) {
                    // TODO: This might not work as planned?
                        tofs_out[pi] = mean_measurements[pi] - 0.5*delays_out[a] - 0.5*delays_out[b];
                    }
                    param_index++; // we only have that here so it
                 }
            }
         }


//         for(int a = 0; a < num_nodes; a++) {
//            for(int b = 0; b < a; b++) {
//
//                size_t pi = pair_index(a, b);
//
//                 if (tof_known[pi]) {
//                    if (tofs_out != NULL) {
//                        tofs_out[pi] = known_tofs[pi];
//                    }
//                 } else {
//                    if (tofs_out != NULL) {
//                        tofs_out[pi] = matrix_data_a[param_index];
//                    }
//                    param_index++;
//                 }
//            }
//         }

         if (param_index != num_input_cols) {
            LOG_ERR("aram_index != num_input_cols, %d, %d", param_index, num_input_cols);
            return;
         }
     }
}


static inline void output_measurement(measurement_t val) {
        char buf[32];
        int64_t int_val = val * 1000.0f;
        snprintf(buf, sizeof(buf), "%lld", int_val);
        uart_out(buf);
}

static void output_measurements(measurement_t *out, size_t num, bool *known) {

    uart_out("[");
    for(size_t i = 0; i < num; i ++) {
        if (known == NULL || known[i]) {
            output_measurement(out[i]);
        } else {
            uart_out("null");
        }

        if (i < num-1) {
            uart_out(", ");
        }
    }
    uart_out("]");
}


static void log_estimation(
    uint8_t num_nodes,
    float32_t delays_out[EST_MAX_NODES],
    float32_t tofs_out[EST_MAX_PAIRS],
    float32_t mean_measurements[EST_MAX_PAIRS],
    float32_t var_measurements[EST_MAX_PAIRS],
    bool delay_known[EST_MAX_NODES],
    float32_t known_delays[EST_MAX_NODES],
    bool tof_known[EST_MAX_PAIRS],
    float32_t known_tofs[EST_MAX_PAIRS]
) {
     uart_out("{ \"mean_measurements\": ");
     output_measurements(mean_measurements, EST_MAX_PAIRS, NULL);
     uart_out("{ \"var_measurements\": ");
     output_measurements(var_measurements, EST_MAX_PAIRS, NULL);

     // we further log the standard deviations
     float32_t std_measurements[EST_MAX_PAIRS];

     for(int i = 0; i < EST_MAX_PAIRS; i++) {
           std_measurements[i] = sqrt(var_measurements[i]);
     }

     uart_out("{ \"std_measurements\": ");
     output_measurements(std_measurements, EST_MAX_PAIRS, NULL);

     uart_out(", \"delays_out\": ");
     output_measurements(delays_out, EST_MAX_NODES, NULL);
     uart_out(", \"tofs_out\": ");
     output_measurements(tofs_out, EST_MAX_PAIRS, NULL);
     uart_out(", \"known_delays\": ");
     output_measurements(known_delays, EST_MAX_NODES, NULL); // we just assume we know everything for now
     uart_out(", \"known_tofs\": ");
     output_measurements(known_tofs, EST_MAX_PAIRS, NULL); // we just assume we know everything for now
        uart_out("}");
}

static void infer_tofs_out(
    uint8_t num_nodes,
    float32_t delays_out[EST_MAX_NODES],
    float32_t tofs_out[EST_MAX_PAIRS],
    float32_t mean_measurements[EST_MAX_PAIRS]
) {
     for(int a = 0; a < num_nodes; a++) {
        for(int b = 0; b < a; b++) {
            size_t pi = pair_index(a, b);
            tofs_out[pi] = mean_measurements[pi] - 0.5*delays_out[a] - 0.5*delays_out[b];
        }
     }
}

void estimate_all(uint16_t round) {

    static measurement_t delays_out[EST_MAX_NODES] = {0.0};
    static measurement_t tofs_out[EST_MAX_PAIRS] = {0.0};
    static measurement_t mean_measurements[EST_MAX_PAIRS] = {0.0};
    static measurement_t var_measurements[EST_MAX_PAIRS] = {0.0};

    static bool delay_known[EST_MAX_NODES];
    static measurement_t known_delays[EST_MAX_NODES];
    static bool tof_known[EST_MAX_PAIRS];
    static measurement_t known_tofs[EST_MAX_PAIRS];

    int64_t estimate_start = 0;
    int64_t estimate_duration = 0;

    //for(int n = EST_MAX_NODES; n <= EST_MAX_NODES; n++)
    int n = EST_MAX_NODES; // we just execute it for a single node right now
    {
        uart_out("{\"type\": \"estimation\", ");
         char buf[32];
         snprintf(buf, sizeof(buf), "\"round\": %hu", round);
         uart_out(buf);

        //initialize known values (measurements, delays, distances
        for(int i = 0; i < n; i++) {
            known_delays[i] = (float32_t) node_factory_antenna_delay_offsets[i];
        }

        for(int p = 0; p < PAIRS(n); p++) {
            known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_TU;
            // TODO: If we reorder, we have to be careful with those distances!
            mean_measurements[p] = get_mean_measurement(p);
            var_measurements[p] = get_var_measurement(p);
        }

        uart_out(", \"mean_measurements\": ");
        output_measurements(mean_measurements, PAIRS(n), NULL);

        uart_out(", \"var_measurements\": ");
        output_measurements(var_measurements, PAIRS(n), NULL);

        // first estimate antenna delays based on all known distances WITHOUT filtering
        {
            memset(tof_known, 1, sizeof(tof_known));
            memset(delay_known, 0, sizeof(delay_known));

            estimate_start = k_uptime_get();
            estimate(
                n,
                delays_out,
                tofs_out,
                mean_measurements,
                var_measurements,
                delay_known,
                known_delays,
                tof_known,
                known_tofs,
                0.0
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"delays_from_measurements\": ");
            output_measurements(delays_out, n, NULL);

            uart_out(", \"delays_from_measurements_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);
        }

        // now use the new delays to infer the distances again -> should result in the same distances basically...
        {
            // we now use the new delay values
            for(int i = 0; i < n; i++) {
                known_delays[i] = delays_out[i];
            }
             for(int p = 0; p < PAIRS(n); p++) {
                // TODO: If we reorder, we have to be careful with those distances!
                mean_measurements[p] = get_mean_measurement(p);
                known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_TU;
            }

            memset(tof_known, 0, sizeof(tof_known));
            memset(delay_known, 1, sizeof(delay_known));

            estimate_start = k_uptime_get();

            infer_tofs_out(
                n,
                delays_out,
                tofs_out,
                mean_measurements
            );

            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"tofs_from_estimated_delays\": ");
            output_measurements(tofs_out, PAIRS(n), NULL);

            uart_out(", \"tofs_from_estimated_delays_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);
        }

        // estimate antenna delays on all known distances WITH filtering
        {
            // reset values just to be sure
            memset(delays_out, 0, sizeof(delays_out));
            memset(tofs_out, 0, sizeof(tofs_out));

            memset(tof_known, 1, sizeof(tof_known));
            memset(delay_known, 0, sizeof(delay_known));

            estimate_start = k_uptime_get();
            estimate(
                n,
                delays_out,
                tofs_out,
                mean_measurements,
                var_measurements,
                delay_known,
                known_delays,
                tof_known,
                known_tofs,
                MAX_VAR_IN_M
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"delays_from_measurements_filtered\": ");
            output_measurements(delays_out, n, NULL);

            uart_out(", \"delays_from_measurements_filtered_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);
        }

        // now use the Filtered delays to infer the distances again -> should result in the same distances basically...
        {
            // we now use the new delay values
            for(int i = 0; i < n; i++) {
                known_delays[i] = delays_out[i];
            }
             for(int p = 0; p < PAIRS(n); p++) {
                // TODO: If we reorder, we have to be careful with those distances!
                mean_measurements[p] = get_mean_measurement(p);
                known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_TU;
            }

            memset(tof_known, 0, sizeof(tof_known));
            memset(delay_known, 1, sizeof(delay_known));

            estimate_start = k_uptime_get();
             infer_tofs_out(
                n,
                delays_out,
                tofs_out,
                mean_measurements
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"tofs_from_filtered_estimated_delays\": ");
            output_measurements(tofs_out, PAIRS(n), NULL);

            uart_out(", \"tofs_from_filtered_estimated_delays_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);
        }


        // now use FACTORY delays to infer the distances again -> should result in the same distances basically...
        {
            // we now use the new delay values
            for(int i = 0; i < n; i++) {
                delays_out[i] = (float32_t) node_factory_antenna_delay_offsets[i];
            }
             for(int p = 0; p < PAIRS(n); p++) {
                // TODO: If we reorder, we have to be careful with those distances!
                mean_measurements[p] = get_mean_measurement(p);
                known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_TU;
            }

            memset(tof_known, 0, sizeof(tof_known));
            memset(delay_known, 1, sizeof(delay_known));

            estimate_start = k_uptime_get();
            infer_tofs_out(
                n,
                delays_out,
                tofs_out,
                mean_measurements
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"tofs_from_factory_delays\": ");
            output_measurements(tofs_out, PAIRS(n), NULL);

            uart_out(", \"tofs_from_factory_delays_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);

        }

        // now use the new delays to infer the distances again -> should result in the same distances basically...
        {
            // we now use the new delay values
            for(int i = 0; i < n; i++) {
                delays_out[i] = 0; // no differences in the delays here
            }
             for(int p = 0; p < PAIRS(n); p++) {
                // TODO: If we reorder, we have to be careful with those distances!
                mean_measurements[p] = get_mean_measurement(p);
                known_tofs[p] = node_distances[p] / SPEED_OF_LIGHT_M_PER_UWB_TU;
            }

            memset(tof_known, 0, sizeof(tof_known));
            memset(delay_known, 1, sizeof(delay_known));

            estimate_start = k_uptime_get();
            infer_tofs_out(
                n,
                delays_out,
                tofs_out,
                mean_measurements
            );
            estimate_duration = k_uptime_delta(&estimate_start);

            uart_out(", \"tofs_uncalibrated\": ");
            output_measurements(tofs_out, PAIRS(n), NULL);

            uart_out(", \"tofs_uncalibrated_time\": ");
            snprintf(buf, sizeof(buf), "%lld", estimate_duration);
            uart_out(buf);
        }
        uart_out("}\n");
    }
}