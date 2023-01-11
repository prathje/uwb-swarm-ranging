
#include <zephyr.h>
#include <stdlib.h>
#include <arm_math_f16.h>
#include <logging/log.h>

#define NUM_NODES 12
#define NUM_PARAMS ((NUM_NODES*(NUM_NODES+1))/2)


LOG_MODULE_REGISTER(matrix_test);

static float16_t matrix_data_a[NUM_PARAMS * NUM_PARAMS];
static float16_t matrix_data_b[NUM_PARAMS * NUM_PARAMS];
static float16_t matrix_data_c[NUM_PARAMS * NUM_PARAMS];


bool matrix_test_f16() {
    LOG_INF("matrix_test_f16");

    int ret = 0;
    arm_matrix_instance_f16 mat_a;
    arm_matrix_instance_f16 mat_b;
    arm_matrix_instance_f16 mat_c;

    // assign memory regions
    {
        mat_a.pData = matrix_data_a; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
        mat_b.pData = matrix_data_b; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));
        mat_c.pData = matrix_data_c; //malloc(NUM_PARAMS * NUM_PARAMS * sizeof(float16_t));


        if (mat_a.pData == NULL || mat_b.pData == NULL || mat_c.pData == NULL ) {
             LOG_INF("Allocation failed...");
            return false;
        }
    }


    // Store X in mat_a
    {
        // TODO: actually store it!
        mat_a.numRows = NUM_PARAMS;
        mat_a.numCols = NUM_PARAMS;
    }

    // transpose X to mat_b
    {
        mat_b.numRows = NUM_PARAMS;
        mat_b.numCols = NUM_PARAMS;

        ret = arm_mat_trans_f16(&mat_a, &mat_b);
         if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... A");
            return false;
        }
    }

    // multiply X^T (mat_b) with X (mat_a) and store in mat_c
    {
        mat_c.numRows = NUM_PARAMS;
        mat_c.numCols = NUM_PARAMS;

        ret = arm_mat_mult_f16(&mat_b, &mat_a, &mat_c);

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

        ret = arm_mat_inverse_f16(&mat_c, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... C");
            return false;
        }
    }

    // multiply (X^T X)^{-1} (mat_a) with X^T (mat_b) and store in mat_c
    {
        mat_c.numRows = NUM_PARAMS;
        mat_c.numCols = NUM_PARAMS;

        ret = arm_mat_mult_f16(&mat_a, &mat_b, &mat_c);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... D");
            return false;
        }
    }


    // load Y into mat_b
    {
        //TODO: implement this
        mat_b.numRows = NUM_PARAMS;
        mat_b.numCols = 1;
    }

    // multiply (X^T X)^{-1}  X^T (mat_c) with Y (mat_b) and store in mat_a
    {
        mat_a.numRows = NUM_PARAMS;
        mat_a.numCols = 1;

        ret = arm_mat_mult_f16(&mat_c, &mat_b, &mat_a);

        if (ret == ARM_MATH_SIZE_MISMATCH ) {
             LOG_INF("SIZE MISMATCH... E");
            return false;
        }
    }
     LOG_INF("Done!!");

    return true;
}

void matrix_test() {

    matrix_test_f16();

}